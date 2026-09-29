import math

import numpy as np
import random
import copy
import heapq
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

# from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree


MAX_TRIAL_SAMPLES = 30 # number of samples to try when sampling grasp and placement poses for boundary points


class Configuration:
    def __init__(self, joint_angles, attachments=None, 
                 is_transition=False, parent_orbit=None, 
                 grasp_poses=None, placement_poses=None,
                 pre_pose=None):
        '''
            Initialize a configuration with a list or tuple of joint angles.

            :param joint_angles: List or tuple of joint angles for the robot(s).
            :param attachments: same as parent orbit's attachments if not specified
            :param is_transition: Boolean indicating if this configuration is a transition point between orbits.
            :param parent_orbit: The parent orbit to which this configuration belongs.
            :param grasp_poses: Dictionary mapping object IDs to grasp poses (4x4 homogeneous transformation matrices).
            :param placement_poses: Dictionary mapping object IDs to placement poses (4x4 homogeneous transformation matrices).
            :param pre_pose: Pre-grasp pose is the pose that robot need to reach before grasping/ placing (list of joint angles).
        '''

        self.joint_angles = joint_angles
        self.attachments = dict(attachments) if attachments is not None else {}
        self.is_transition = bool(is_transition)
        # PRM may clone one physical-looking transition into separate graph
        # states for different incident-orbit contexts.  This is assigned
        # before the node is inserted into a hash-based roadmap structure.
        self.roadmap_context = None

        if parent_orbit is not None and isinstance(parent_orbit, Orbit):
            self.parent_orbit = [parent_orbit]
        elif parent_orbit is not None and isinstance(parent_orbit, list) and len(parent_orbit) >= 1:
            self.parent_orbit = parent_orbit
        else:
            self.parent_orbit = None

        if not is_transition and (grasp_poses is not None or placement_poses is not None or pre_pose is not None):
            raise ValueError("Non-transition configurations should not have grasp or placement poses defined.")

        if is_transition and (grasp_poses is None and placement_poses is None and pre_pose is None):
            raise ValueError("Transition configurations must have grasp or placement poses defined.")
        else:
            self.grasp_poses: Dict[int, np.ndarray] = grasp_poses if grasp_poses else {}                # object_id -> grasp pose T_grasp (4x4 homogeneous transformation matrix)
            self.placement_poses: Dict[int, np.ndarray] = placement_poses if placement_poses else {}    # object_id -> T_world (when not grasped)
            self.pre_pose: list = pre_pose if pre_pose is not None else []                              # list of joint angles

        if not is_transition and self.parent_orbit and attachments is None:
            self.attachments = dict(self.parent_orbit[0].attachments)


    def add_parent_orbit(self, orbit):
        if self.parent_orbit is None:
            self.parent_orbit = [orbit]
        elif not any(parent is orbit for parent in self.parent_orbit):
            self.parent_orbit.append(orbit)

    def _parent_orbit_identity_key(self):
        # Parent-orbit identity is only part of non-transition node identity.
        if self.is_transition or not self.parent_orbit:
            return None
        return tuple(sorted(id(orbit) for orbit in self.parent_orbit))

    def is_valid(self, environment):
        return environment.is_valid(self)
    
    def __eq__(self, other):
        if not isinstance(other, Configuration):
            return False
        return (
            self.is_transition == other.is_transition and
            (
                not self.is_transition
                or self.roadmap_context == other.roadmap_context
            ) and
            self._parent_orbit_identity_key() == other._parent_orbit_identity_key() and
            _deep_equal(self.joint_angles, other.joint_angles) and
            _deep_equal(self.attachments, other.attachments) and
            _deep_equal(self.grasp_poses, other.grasp_poses) and
            _deep_equal(self.placement_poses, other.placement_poses) and 
            _deep_equal(self.pre_pose, other.pre_pose)
        )

    def __hash__(self):
        return hash((
            self.is_transition,
            self.roadmap_context if self.is_transition else None,
            self._parent_orbit_identity_key(),
            _to_hashable(self.joint_angles),
            _to_hashable(self.attachments),
            _to_hashable(self.grasp_poses),
            _to_hashable(self.placement_poses),
            _to_hashable(self.pre_pose),
        ))



class RobotState(Enum):
    TRANSIT = auto()
    TRANSFER = auto()

class ObjectState(Enum):
    TRANSIT = auto()
    TRANSFER = auto()


@dataclass(frozen=True, slots=True)
class Mode:
    """Canonical operational constraints defined only by attachments.

    Attachment values are ordered by robot ID. Exact poses, collision state,
    task progress, and geometric connectivity deliberately do not participate
    in mode identity.
    """

    attachments: tuple[Optional[int], ...]

    def __post_init__(self):
        object.__setattr__(self, "attachments", tuple(self.attachments))

    @classmethod
    def from_attachments(
        cls,
        attachments: Mapping[int, Optional[int]] | Sequence[Optional[int]],
        robot_ids: Optional[Sequence[int]] = None,
    ):
        """Build a mode signature independent of dictionary insertion order."""
        if isinstance(attachments, Mapping):
            ordered_robot_ids = (
                tuple(sorted(attachments))
                if robot_ids is None
                else tuple(sorted(robot_ids))
            )
            values = tuple(attachments.get(robot_id) for robot_id in ordered_robot_ids)
        else:
            values = tuple(attachments)
        return cls(values)

    @classmethod
    def from_orbit(cls, orbit, robot_ids: Optional[Sequence[int]] = None):
        return cls.from_attachments(orbit.attachments, robot_ids=robot_ids)


@dataclass(frozen=True, slots=True)
class ModalState:
    """High-level search state: operational mode plus task progress."""

    mode: Mode
    at_goal: tuple[bool, ...]

    def __post_init__(self):
        object.__setattr__(self, "at_goal", tuple(bool(value) for value in self.at_goal))


@dataclass(frozen=True, slots=True)
class ModalAction:
    """A discrete pick or place action labeling a directed modal edge."""

    action_type: str
    robot_id: int
    object_id: int

    def __post_init__(self):
        normalized_action = self.action_type.lower()
        if normalized_action not in {"pick", "place"}:
            raise ValueError("Modal action must be either 'pick' or 'place'.")
        object.__setattr__(self, "action_type", normalized_action)


class Orbit:
    def __init__(self, configuration, neighboring_orbit=None, init_states=None, environment=None) -> None:

        '''
            :param configuration: can be an interior or boundary point
            :param neighboring_orbit: an optional neighboring orbit that shares a boundary with this orbit (used to define the transition between orbits)
            :param init_states: the orbit can be defined from its neigboring orbit or specified initial states of robots and objects
            :param environment: the environment used for checking feasibility and goal condition
        '''

        self.environment = environment

        # Assigned by ModalGraph only after OrbitGraph has canonicalized this
        # geometric orbit. These references do not participate in orbit
        # equality or hashing.
        self.parent_mode: Optional[Mode] = None
        self.parent_modal_state: Optional[ModalState] = None

        # Legacy symbolic fields remain orbit data for existing environments,
        # transition construction, and saved-graph diagnostics.
        self.robot_states: Dict[int, RobotState] = {}
        self.object_states: Dict[int, ObjectState] = {}
        self.attachments: Dict[int, Optional[int]] = {}

        self.r_min = environment.r_min if environment is not None else 0.1
        self.r_prm = environment.r_prm if environment is not None else 1.0
        self.k0_prm = environment.k_prm if environment is not None else 5
        self.embed_dim = environment.config_dim


        # Orbit level information (Dictionaries mapping ID -> Pose)
        self.grasp_poses: Dict[int, np.ndarray] = {}            # object_id -> grasp pose T_grasp (4x4 homogeneous transformation matrix)
        self.placement_poses: Dict[int, np.ndarray] = {}        # object_id -> T_world (when not grasped)

        self.configurations = [configuration]
        self.update_kd_tree()    

        self.is_goal_orbit = False  
        self.goal_configurations = []  # list of configurations that satisfy the goal condition

        # heuristic value for the orbit, used in drrt* to guide the search towards the goal orbit
        self.heuristic_value = float('inf')  # Initialize with infinity


        # The orbit is defined by transition points and neighboring orbit
        if neighboring_orbit is not None:
            self.define_orbit_by_neighbor(configuration, neighboring_orbit)

        # The orbit is defined by initial states of robots and objects
        elif init_states is not None:
            self.robot_states = init_states['robot_states']
            self.object_states = init_states['object_states']   
            self.attachments = init_states['attachments']
            self.grasp_poses = init_states['grasp_poses']
            self.placement_poses = init_states['placement_poses']   
        else:
            raise ValueError("Either neighboring_orbit or initial states must be provided to define the orbit.")

    def __eq__(self, other):
        if not isinstance(other, Orbit):
            return False
        return (
            _deep_equal(self.robot_states, other.robot_states) and
            _deep_equal(self.object_states, other.object_states) and
            _deep_equal(self.attachments, other.attachments) and
            _deep_equal(self.grasp_poses, other.grasp_poses) and
            _deep_equal(self.placement_poses, other.placement_poses)
        )
    
    def __hash__(self):
        return hash((
            _to_hashable(self.robot_states),
            _to_hashable(self.object_states),
            _to_hashable(self.attachments),
            _to_hashable(self.grasp_poses),
            _to_hashable(self.placement_poses),
        ))


    @property
    def r(self):

        # Standard PRM* with r = gamma * (log(n)/n)^(1/d)
        n = max(len(self.configurations), 2)
        scaling_factor = 1.0  # You can adjust this factor to make the radius larger or smaller
        radius_scaling = scaling_factor * ((math.log(n) / n) ** (1.0 / self.embed_dim))
        # prm_star_radius = self.r_prm * radius_scaling

        # uncomment this if want to use PRM
        prm_star_radius = self.r_prm

        return max(self.r_min, prm_star_radius)
    
    @property
    def k(self):
        # Standard PRM* with k = k * log(n)
        n = max(len(self.configurations), 2)

        k_prm = max(int(self.k0_prm * math.log(n)), self.k0_prm)

        return k_prm


    def define_orbit_by_neighbor(self, transition_configuration: Configuration, neighboring_orbit):
        '''
            Define this orbit by a transition configuration and a neighboring orbit that shares a boundary with this orbit.
        '''

        if transition_configuration.is_transition == False:
            raise ValueError("The new orbit must be defined by a transition configuration that lies on the boundary between two orbits.")

        # first copy object states then change based on the transition
        self.robot_states = copy.deepcopy(neighboring_orbit.robot_states)
        self.object_states = copy.deepcopy(neighboring_orbit.object_states)
        self.attachments = copy.deepcopy(neighboring_orbit.attachments)
        self.grasp_poses = copy.deepcopy(neighboring_orbit.grasp_poses)
        self.placement_poses = copy.deepcopy(neighboring_orbit.placement_poses)
        
        # Update orbit-level information
        for r_id, r_state in neighboring_orbit.robot_states.items():
            # -----------------------------
            # CASE 1: PICK (Transit → Transfer)
            # -----------------------------
            if r_state == RobotState.TRANSIT and transition_configuration.attachments[r_id] is not None:
                self.robot_states[r_id] = RobotState.TRANSFER
                obj_id = next(
                    (obj for rid, obj in transition_configuration.attachments.items() if rid == r_id),
                    None
                )
                self.object_states[obj_id] = ObjectState.TRANSFER
                self.attachments[r_id] = obj_id
                if self.placement_poses.get(obj_id) is not None:
                    del self.placement_poses[obj_id]        # object is no longer at placement pose
                    # otherwise this can be the handover action
                self.grasp_poses[obj_id] = transition_configuration.grasp_poses[obj_id]  # update grasp pose for the object

            # -----------------------------
            # CASE 2: PLACE (Transfer → Transit)
            # -----------------------------
            elif r_state == RobotState.TRANSFER and transition_configuration.attachments[r_id] is None:
                self.robot_states[r_id] = RobotState.TRANSIT
                obj_id = next(
                    (obj for rid, obj in neighboring_orbit.attachments.items() if rid == r_id),
                    None
                )
                self.object_states[obj_id] = ObjectState.TRANSIT
                self.attachments[r_id] = None          # object is no longer held by the robot
                if transition_configuration.placement_poses.get(obj_id) is not None:
                    self.placement_poses[obj_id] = transition_configuration.placement_poses[obj_id]  # update placement pose for the object
                    del self.grasp_poses[obj_id]              # object is no longer held by the robot
                else:
                    # otherwise this can be the handover action
                    # the grasp pose is updated in the pick action of the other robot.
                    self.object_states[obj_id] = ObjectState.TRANSFER  # still held by another robot

            # -----------------------------
            # CASE 3: no change
            # -----------------------------
            else:
                continue    # nothing changes for this robot or the objects it holds

    

    def sample_interior_points(self, num_points, environment):
        '''
            Sample a set of continuous configurations that lie in the interior of this orbit.
        '''
        interior_points = []

        i=0
        trial_count = 0
        while i < num_points and trial_count < MAX_TRIAL_SAMPLES:
            point = environment.sample_interior_configuration(self)
            trial_count += 1
            if point is not None:
                interior_points.append(point)
                i += 1
        self.configurations.extend(interior_points)
        self.update_kd_tree()
        return interior_points
    

    def sample_boundary_points(
        self,
        num_points,
        environment,
        desired_next_orbit=None,
        desired_modal_state=None,
        modal_graph=None,
        modal_transition_bias=0.8,
    ):
        '''
            Sample a set of continuous configurations that lie on the boundary of this orbit.

            Orbit-path guidance takes priority over modal-path guidance.  For
            boundary resampling, the next orbit contributes its modal target
            rather than requiring an exact match of continuous grasp and
            placement poses.
            Nonmatching valid transitions retain probability
            ``1 - modal_transition_bias`` and remain discoverable.
        '''
        boundary_points = []

        i=0
        trial_count = 0
        while i < num_points and trial_count < MAX_TRIAL_SAMPLES:
            point = environment.sample_boundary_configuration(self)
            trial_count += 1
            if point is not None:
                matches_guidance = True
                if desired_next_orbit is not None:
                    target_state = desired_next_orbit.parent_modal_state
                    matches_guidance = (
                        target_state is not None
                        and modal_graph is not None
                        and modal_graph.transition_advances_toward(
                            self,
                            point,
                            target_state,
                        )
                    )
                elif desired_modal_state is not None and modal_graph is not None:
                    matches_guidance = modal_graph.transition_advances_toward(
                            self,
                            point,
                            desired_modal_state,
                    )
                if (
                    not matches_guidance
                    and random.random() < modal_transition_bias
                ):
                    continue
                boundary_points.append(point)
                i += 1
        self.configurations.extend(boundary_points)
        self.update_kd_tree()
        return boundary_points


    def define_neighboring_orbit(self, boundary_configuration: Configuration):
        '''
            Define a neighboring orbit that shares a boundary with this orbit based on the given boundary configuration.
        '''
        return Orbit(configuration=boundary_configuration, neighboring_orbit=self, environment=self.environment)


    def update_kd_tree(self):
        if not self.configurations:
            self.k_d_tree = None
            return

        # if self.radian_searching:
        #     search_coords = np.array([
        #         embed_angles(cfg.joint_angles)
        #         for cfg in self.configurations
        #     ])
        # else:
        search_coords = np.array([
            cfg.joint_angles
            for cfg in self.configurations
        ])
        search_coords = search_coords.reshape(len(self.configurations), -1)
        # self.k_d_tree = KDTree(search_coords, metric='euclidean')
        self.k_d_tree = cKDTree(search_coords)
    
    
    def nearest_neighbors(self, configuration: Configuration) -> list[Configuration]:
        '''
            Find the nearest neighbors of a given configuration in this orbit.
        '''
        return self.k_nearest_neighbors(configuration)
        
    

    def radius_nearest_neighbors(self, configuration: Configuration) -> list[Configuration]:
        '''
            Find all neighbors of a given configuration in this orbit within a connection radius r.
        '''
        if not self.configurations:
            return []

        query_coords = np.array(configuration.joint_angles).reshape(1, -1)

        # Use query_ball_point to find all indices within distance 'r'
        indices = self.k_d_tree.query_ball_point(query_coords, r=self.r)[0]

        # Handle the edge case where no neighbors are found within the radius
        if not indices:
            return []

        # Map the discovered indices back to your Configuration instances
        neighbors = [self.configurations[i] for i in indices]

        return neighbors
    

    def k_nearest_neighbors(self, configuration: Configuration) -> list[Configuration]:
        '''
            Find the k nearest neighbors of a given configuration in this orbit.
        '''
        if not self.configurations:
            return []

         # Flattens 2x7 directly into a 14-element 1D array to match tree dimensions
        query_coords = np.array(configuration.joint_angles).flatten()

        # Use query to find the k nearest neighbors
        distances, indices = self.k_d_tree.query(query_coords, k=min(self.k, len(self.configurations)))

        indices_arr = np.asarray(indices, dtype=int).flatten()

        if indices_arr.size == 0:
            return []


        # Map the discovered indices back to your Configuration instances
        neighbors = [self.configurations[i] for i in indices_arr]

        return neighbors
        


    def is_goal(self, environment) -> bool:
        '''
            Check if this orbit satisfies the goal condition.
        '''
        is_goal_now = environment.is_goal_orbit(self)
        self.is_goal_orbit = bool(is_goal_now)
        return self.is_goal_orbit
    
    def find_goal_configuration(self, configuration: Configuration) -> Optional[Configuration]:
        """
        Scans the list for a matching configuration.
        Returns the original stored object reference if found, otherwise None.
        """
        for existing_config in self.goal_configurations:
            if existing_config == configuration:
                return existing_config
        return None
    

    def add_goal_configuration(self, configuration: Configuration) -> Configuration:
        """
        Add a configuration that satisfies the goal condition to the list of goal configurations.
        """
        if not configuration.is_transition and self.is_goal_orbit:
            # Call the finder exactly ONCE and store the result
            existing_config = self.find_goal_configuration(configuration)
            if existing_config is not None:
                return existing_config  # Already exists, return the original instance
            
            # If it doesn't exist, safely append it to the list
            self.goal_configurations.append(configuration)
            
        return configuration
    

def embed_angles(all_joint_angles):
    embedded = []

    for robot_angles in all_joint_angles:
        for theta in robot_angles:
            embedded.extend([
                np.sin(theta),
                np.cos(theta)
            ])

    return embedded


def dijkstra(roadmap_graph, start_node, target_node=None):
    """
        Finds the shortest path in a RoadmapGraph.
        'roadmap_graph' is a dict of dicts:
        {Configuration: {Configuration: distance, ...}, ...}

        Returns (distances, parent_map). Use reconstruct_path() to get the path.
            distances: dict mapping each node to its distance from start_node
            parent_map: dict mapping each node to its parent node in the shortest path tree
    """
    distances = {node: float('inf') for node in roadmap_graph}
    distances[start_node] = 0

    # Use a counter as tiebreaker so heapq never compares Configuration objects
    counter = 0
    priority_queue = [(0, counter, start_node)]     # cost, counter, node

    parent_map = {start_node: None}

    while priority_queue:
        current_cost, _, current_node = heapq.heappop(priority_queue)   # choose the node with the lowest cost

        if current_node is target_node:
            return distances, parent_map

        if current_cost > distances[current_node]:
            continue

        for neighbor, edge_weight in roadmap_graph[current_node].items():
            new_cost = current_cost + edge_weight

            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost      # update cost
                parent_map[neighbor] = current_node # update parent
                counter += 1
                heapq.heappush(priority_queue, (new_cost, counter, neighbor))

    return distances, parent_map


def reconstruct_path(parent_map, start_node, goal_node):
    """
        Reconstruct the shortest path from start_node to goal_node
        using the parent_map returned by dijkstra().

        :return: List of nodes representing the path from start_node to goal_node, or None if no path exists.
    """
    if goal_node not in parent_map:
        return None
    path = []
    node = goal_node
    while node is not None:
        path.append(node)
        node = parent_map[node]
    path.reverse()
    return path


# === Ultils for transformations between different coordinate frames ===

    

# === Utils for hashing and equality of complex data structures ===
def _deep_equal(a, b, atol=1e-6):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol))
        except Exception:
            # Fallback when values are not numeric arrays.
            return bool(np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object)))

    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_deep_equal(a[k], b[k], atol=atol) for k in a)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y, atol=atol) for x, y in zip(a, b))

    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return abs(float(a) - float(b)) <= atol

    return a == b


def _to_hashable(val):
    if isinstance(val, np.ndarray):
        rounded = np.round(val, 5)
        return ("ndarray", tuple(_to_hashable(x) for x in rounded.tolist()))

    if isinstance(val, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in val.items()))

    if isinstance(val, (list, tuple)):
        return tuple(_to_hashable(v) for v in val)

    if isinstance(val, (float, np.floating)):
        return round(float(val), 5)

    return val


def freeze(d):
    return _to_hashable(d)
