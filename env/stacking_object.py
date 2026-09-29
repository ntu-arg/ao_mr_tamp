import os
import shutil
import ast
import re
import sys

import pybullet as p
import pybullet_data
import numpy as np
from collections import defaultdict
import time
import copy
import random
from scipy.spatial.transform import Rotation as R

from env.env_utils import get_all_subsets, interpolate_motion
from planners.prm_roadmap_utils import Configuration, RobotState, ObjectState
from env.simple_pick_place import SimplePickPlaceEnv, MovableObject, FrankaPandaRobot, FR3_LIMITS, \
    sample_grasp_pose, sample_placement_pose, sample_yaw_quaternion



SAFE_APPROACH_DISTANCE = 0.005                  # for collision checking: all parts should keep 1,5cm distance from each other for safe 
ALLOWED_PLACEMENT_CONTACT_DISTANCE = 0.025      # allow small contact distance for placed objects with the table (for stability)

STACKING_APPROACH_DISTANCE = 0.01               # allow 1cm height difference for stacking objects


#------------------------------
#----- FOR DEBUGGING ONLY -----

print_debug = False

#----------- END ---------


class StackingObjectsEnv(SimplePickPlaceEnv):
    """
        The environment of two robots performing a stacking objects task.
        Using robot: Franka Panda

        Searching space for robot: 
        [
            [joint1_r1, joint2_r1, ..., joint7_r1], # Robot 1 state
            [joint1_r2, joint2_r2, ..., joint7_r2]  # Robot 2 state
        ]

    """

    def __init__(self, num_robots=1, num_mobjects=1, show_gui=True):

        if show_gui:
            self.using_client = p.connect(p.GUI)
        else:
            self.using_client = p.connect(p.DIRECT)

        self.num_robots = num_robots
        self.num_mobjects = num_mobjects

        self.r_min = 1.0        # PRM lower bound connection threshold
        self.r_prm = 18.0       # PRM base scaling factor (gamma) for maximum initial search radius
        self.k_prm = 10          # PRM number of nearest neighbors to connect to

        # collision checking parameters
        # This is simple environment so we use coarse values of about 10 degrees for each joint.
        self.collision_checking_resolution = 0.5  # radian, changind depends on the complexity of evironment.
        self.collision_joint_resolution = 0.2     # radian

        self.robot_list = list(range(num_robots))
        self.transition_robot_subsets = get_all_subsets(self.robot_list)

        self.robots = []
        self.mobjects = []
        self.plane = None


        self.init_robot_states = {}
        self.init_object_states = {}
        self.init_attachments = {}          # No attachments at the start
        self.init_grasp_poses = {}          # No grasping poses at the start
        self.init_placement_poses = {}      # Init position of objects can be used as placement poses

        self.config_dim = self.num_robots * 7    # 7 joint angles for each robot

        self.joint_limits = FR3_LIMITS
        self.debugger_attached = sys.gettrace() is not None

        self.reset(client_id=self.using_client)

    @staticmethod
    def _alternating_object_y_positions(
        num_objects,
        min_y=-0.5,
        max_y=0.5,
    ):
        """Distribute object IDs alternately between the two robot sides."""
        if num_objects <= 0:
            return []
        if num_objects == 1:
            return [0.0]

        available_positions = np.linspace(min_y, max_y, num_objects)
        alternating_positions = []
        lower_index = 0
        upper_index = num_objects - 1

        while lower_index <= upper_index:
            alternating_positions.append(
                float(available_positions[lower_index])
            )
            lower_index += 1
            if lower_index <= upper_index:
                alternating_positions.append(
                    float(available_positions[upper_index])
                )
                upper_index -= 1

        return alternating_positions


    # ========================
    # RESET
    # ========================
    def reset(self, client_id=None):
        if client_id is None:
            client_id = self.using_client
        # Keep all subsequent physics queries/actions on the same world as the latest reset.
        self.using_client = client_id
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.using_client)
        p.resetSimulation(physicsClientId=self.using_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.using_client)

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.using_client)

        # Table
        table1_pos = [0.5, 0.0, 0.0]
        table = p.loadURDF(
            "models/custom_table/custom_table.urdf", 
            table1_pos, 
            p.getQuaternionFromEuler([0, 0, np.pi/2]),
            useFixedBase=True,
            physicsClientId=self.using_client
        )
        self.fixed_objects = [table]
        table_height = 0.88


        # Robot base — mounted on the table surface (table z + table height=0.625)
        self.robots = []
        table_surface_z = table1_pos[2] + table_height  
        robot_base_z = 0.02
        offset_y = [0.5, -0.5]  
        for i in range(self.num_robots):
            rid = i
            offset_y_i = offset_y[i % len(offset_y)]
            offset_x_i = -0.2
            robot_base_pos = [table1_pos[0] + offset_x_i, table1_pos[1] + offset_y_i, table_surface_z + robot_base_z]  # on the table surface
            robot = FrankaPandaRobot(id=rid, base_pos=robot_base_pos, client_id=self.using_client)
            self.robots.append(robot)

            self.init_attachments[rid] = None                   # No attachments at the start 
            self.init_robot_states[rid] = RobotState.TRANSIT    # All robots start in TRANSIT state


        # Create multiple movable objects
        self.mobjects = []
        object_y_positions = self._alternating_object_y_positions(
            self.num_mobjects
        )
        for i in range(self.num_mobjects):
            oid = i
            offset_x = 0.1
            offset_y = object_y_positions[i]
            offset_z = 0.02   # from the center of the cube to the table surface
            obj_pos = [table1_pos[0] + offset_x, table1_pos[1] + offset_y, table_surface_z + offset_z]  # on the table surface
            obj_orientation = p.getQuaternionFromEuler([0, 0, 0])   # no rotation

            offset_x_goal = 0.3
            offset_y_goal = 0.0
            offset_z_goal = 0.02 + 0.0501*i   # from the center of the cube to the table surface, stack them with 5cm height difference
            target_pos = [table1_pos[0] + offset_x_goal, table1_pos[1] + offset_y_goal, table_surface_z + offset_z_goal]  # on the table surface

            obj = MovableObject(
                obj_pos, 
                obj_orientation,
                x_goal=target_pos[0], y_goal=target_pos[1], z_goal=target_pos[2], 
                id=oid, sim_id=None,
                client_id=self.using_client
            )

            # init pose
            obj_pose = np.eye(4)
            obj_pose[:3, 3] = obj_pos
            obj_pose[:3, :3] = np.array(p.getMatrixFromQuaternion(obj_orientation)).reshape(3, 3)
            self.init_placement_poses[oid] = obj_pose    
            self.mobjects.append(obj)

            self.init_object_states[oid] = ObjectState.TRANSIT  # All objects start at being free


        # Let newly spawned objects settle under gravity, then store the true stable poses.
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.using_client)

        for obj in self.mobjects:
            settled_pos, settled_orn = p.getBasePositionAndOrientation(
                obj.sim_id, physicsClientId=self.using_client
            )
            obj.x, obj.y, obj.z = [round(val, 3) for val in settled_pos]

            obj_pose = np.eye(4)
            obj_pose[:3, 3] = np.array([obj.x, obj.y, obj.z])
            obj_pose[:3, :3] = np.array(p.getMatrixFromQuaternion(settled_orn)).reshape(3, 3)
            self.init_placement_poses[obj.id] = obj_pose

            print("Created one movable object with id {}, initial position {}, and goal regions {}".format(
                obj.id, [obj.x, obj.y, obj.z], obj.get_goal_position()
            ))


        self._draw_object_targets(client_id=self.using_client)  # Visualize target objects in the same client

        # Turn of all the coordinates
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        # # FOR CHECKING GRASPING POSES
        # state = p.getLinkState(self.robots[0].sim_id, self.robots[0].ee_link_index, physicsClientId=self.using_client)
        # pos = state[4]
        # orn = state[5]
        # print(f"EE pose: pos {pos}, orn {orn}")
        # self.draw_frame(pos, orn)

        # state = p.getLinkState(self.robots[1].sim_id, self.robots[1].ee_link_index, physicsClientId=self.using_client)
        # pos = state[4]
        # orn = state[5]
        # print(f"EE pose: pos {pos}, orn {orn}")
        # self.draw_frame(pos, orn)

        # obj_state = p.getBasePositionAndOrientation(self.mobjects[0].sim_id, physicsClientId=self.using_client)
        # obj_pos = obj_state[0]
        # obj_orn = obj_state[1]
        # self.draw_frame(obj_pos, obj_orn)




    def _has_forbidden_collision(self, all_robot_sim_ids, attachment_sim_ids, placed_objects_sim_ids, stacking_pairs=None, debug_context=""):
        """Run layered collision checks and return True when any forbidden collision is found."""
        p.performCollisionDetection(physicsClientId=self.using_client)

        # ``stacking_action_check`` reports logical object IDs, whereas all
        # collision pairs below use PyBullet body IDs.  Convert once here so
        # legitimate contact between stacked objects is actually allowed.
        stacking_body_pairs = {
            (
                self.mobjects[bottom_id].sim_id,
                self.mobjects[top_id].sim_id,
            )
            for bottom_id, top_id in (stacking_pairs or ())
        }

        robot_ids = list(all_robot_sim_ids)
        fixed_object_ids = list(self.fixed_objects)
        placed_ids = set(placed_objects_sim_ids)
        attached_pairs = {
            r_sim: o_sim
            for r_sim, o_sim in attachment_sim_ids.items()
            if o_sim is not None
        }
        attached_obj_ids = set(attached_pairs.values())
        movable_obj_ids = sorted(placed_ids | attached_obj_ids) # set union

        fingertip_links_by_robot = {
            robot.sim_id: set(robot.fingertip_link_indices)
            for robot in self.robots
        }

        def _debug_collision(contact, prefix="Collision detected"):
            if not print_debug:
                return
            bodyA = contact[1]
            bodyB = contact[2]
            contact_distance = contact[8]

            bodyA_name = None
            bodyB_name = None
            for r_id, robot in enumerate(self.robots):
                if bodyA == robot.sim_id:
                    bodyA_name = f"Robot{r_id}"
                if bodyB == robot.sim_id:
                    bodyB_name = f"Robot{r_id}"
            for o_id, obj in enumerate(self.mobjects):
                if bodyA == obj.sim_id:
                    bodyA_name = f"Object{o_id}"
                if bodyB == obj.sim_id:
                    bodyB_name = f"Object{o_id}"
            if bodyA in self.fixed_objects:
                bodyA_name = f"Table{bodyA}"
            if bodyB in self.fixed_objects:
                bodyB_name = f"Table{bodyB}"
            if bodyA == self.plane:
                bodyA_name = f"Plane{bodyA}"
            if bodyB == self.plane:
                bodyB_name = f"Plane{bodyB}"
            if bodyA_name is None:
                bodyA_name = f"Body{bodyA}"
            if bodyB_name is None:
                bodyB_name = f"Body{bodyB}"

            if debug_context:
                print(f"In {debug_context} checking:: {prefix} between {bodyA_name} and {bodyB_name} with contact distance {contact_distance}")
            else:
                print(f"{prefix} between {bodyA_name} and {bodyB_name} with contact distance {contact_distance}")
            print(f"Contact details: {contact}")

        def _first_penetration_contacts(body_a, body_b, max_distance=0.0):
            contacts = p.getClosestPoints(
                bodyA=body_a,
                bodyB=body_b,
                distance=max_distance,               # ignore points of distance > max_distance
                physicsClientId=self.using_client,
            )
            return [c for c in contacts if c[8] <= 0]

        # LAYER 2: ATTACHED OBJECTS INTERACTIONS
        # Layer 2a: attached object against its owning robot (allow fingertip links only).
        for robot_sim_id, attached_obj_sim_id in attached_pairs.items():
            for c in _first_penetration_contacts(robot_sim_id, attached_obj_sim_id, max_distance=SAFE_APPROACH_DISTANCE):
                link_robot = c[3] if c[1] == robot_sim_id else c[4]
                if link_robot in fingertip_links_by_robot.get(robot_sim_id, set()):
                    continue
                _debug_collision(c, prefix="Forbidden attachment collision")
                return True

        # Layer 2b: movable objects (placed and attached) with fixed objects.
        for obj_sim_id in movable_obj_ids:
            for fixed_sim_id in fixed_object_ids:
                contacts = _first_penetration_contacts(obj_sim_id, fixed_sim_id, max_distance=SAFE_APPROACH_DISTANCE)
                for c in contacts:
                    if obj_sim_id in placed_ids and c[8] >= -ALLOWED_PLACEMENT_CONTACT_DISTANCE:
                        continue
                    if c[8] <= 0:
                        _debug_collision(c, prefix="Object-environment collision")
                        return True
            # with plane
            for c in _first_penetration_contacts(obj_sim_id, self.plane, max_distance=SAFE_APPROACH_DISTANCE):
                if obj_sim_id in placed_ids and c[8] >= -ALLOWED_PLACEMENT_CONTACT_DISTANCE:
                    continue
                if c[8] <= 0:
                    _debug_collision(c, prefix="Object-plane collision")
                    return True

        # Layer 2c: movable objects with each other.
        movable_obj_list = sorted(movable_obj_ids)
        for i in range(len(movable_obj_list)):
            for j in range(i + 1, len(movable_obj_list)):
                if (
                    (movable_obj_list[i], movable_obj_list[j]) in stacking_body_pairs
                    or (movable_obj_list[j], movable_obj_list[i]) in stacking_body_pairs
                ):

                    continue    # allow contact of stacking pairs 
                for c in _first_penetration_contacts(movable_obj_list[i], movable_obj_list[j], max_distance=0.0):
                    _debug_collision(c, prefix="Object-object collision")
                    return True


        # LAYER 3: ROBOT INTERACTIONS
        # Layer 3a: robot-robot collisions.
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                for c in _first_penetration_contacts(robot_ids[i], robot_ids[j], max_distance=SAFE_APPROACH_DISTANCE):
                    _debug_collision(c, prefix="Robot-robot collision")
                    return True

        # Layer 3b: robot with fixed objects, allowing base/table support contacts.
        for robot_sim_id in robot_ids:
            robot_base_links = set(self.robots[robot_ids.index(robot_sim_id)].base_link_index)
            for fixed_sim_id in fixed_object_ids:
                for c in _first_penetration_contacts(robot_sim_id, fixed_sim_id, max_distance=SAFE_APPROACH_DISTANCE):
                    link_robot = c[3] if c[1] == robot_sim_id else c[4]
                    if link_robot in robot_base_links:
                        continue
                    _debug_collision(c, prefix="Robot-environment collision")
                    return True
            # robot with plane
            for c in _first_penetration_contacts(robot_sim_id, self.plane, max_distance=SAFE_APPROACH_DISTANCE):
                _debug_collision(c, prefix="Robot-plane collision")
                return True

        # Layer 3c: robot with movable objects.
        for robot_sim_id in robot_ids:
            owner_obj_sim_id = attached_pairs.get(robot_sim_id)
            for obj_sim_id in movable_obj_ids:
                for c in _first_penetration_contacts(robot_sim_id, obj_sim_id, max_distance=SAFE_APPROACH_DISTANCE):
                    if owner_obj_sim_id == obj_sim_id:
                        link_robot = c[3] if c[1] == robot_sim_id else c[4]
                        if link_robot in fingertip_links_by_robot.get(robot_sim_id, set()):
                            continue
                    _debug_collision(c, prefix="Robot-object collision")
                    return True
        return False
    

    def stacking_action_check(self, configuration, orbit):
        """Evaluate stack stability without mutating later planning checks."""
        state_id = p.saveState(physicsClientId=self.using_client)
        try:
            return self._stacking_action_check_in_saved_state(
                configuration,
                orbit,
            )
        finally:
            p.restoreState(
                stateId=state_id,
                physicsClientId=self.using_client,
            )
            p.removeState(state_id, physicsClientId=self.using_client)

    def _stacking_action_check_in_saved_state(self, configuration, orbit):
        """
        Check if the given configuration is feasible for stacking action 
        and extracts the tracking pairs of stacked objects.
        """
        # Place objects in their target positions
        for object_id, pose in orbit.placement_poses.items():
            self.mobjects[object_id].set_pose(pose)
        
        for object_id, pose in configuration.placement_poses.items():
            self.mobjects[object_id].set_pose(pose)

        # Settle physics
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.using_client)

        object_positions = {}
        footprint_map = {}  # Initialize the map before using it (x,y): (obj_id, z)
        for obj in self.mobjects:
            settled_pos, settled_orn = p.getBasePositionAndOrientation(
                obj.sim_id, physicsClientId=self.using_client
            )
            # Extract target position from the transformation matrices safely
            if obj.id in configuration.placement_poses:
                target_pos = configuration.placement_poses[obj.id][:3, 3]
            elif obj.id in orbit.placement_poses:
                target_pos = orbit.placement_poses[obj.id][:3, 3]
            else:
                continue # Skip objects not involved in this milestone sequence
            # Stability Check: Reject immediately if the object slipped/fell
            if not np.allclose(settled_pos, target_pos, atol=ALLOWED_PLACEMENT_CONTACT_DISTANCE):
                return False, []
            # Round position to 2 decimal places to prevent floating point noise issues
            settled_pos_rounded = tuple(round(coord, 2) for coord in settled_pos)
            object_positions[obj.id] = settled_pos_rounded
            # Saving the footprint map
            coord_key = (settled_pos_rounded[0], settled_pos_rounded[1])
            current_z = settled_pos_rounded[2]
            if coord_key not in footprint_map:
                footprint_map[coord_key] = [(obj.id, current_z)]
            else:
                footprint_map[coord_key].append((obj.id, current_z))

        # Identify stacking pairs based on the footprint map and z-coordinates
        stacking_pairs = set()
        for object_id, pos in object_positions.items():
            coord_key = (pos[0], pos[1])
            if coord_key in footprint_map:
                for other_obj_id, other_z in footprint_map[coord_key]:
                    if other_obj_id == object_id:
                        continue
                    if np.isclose(pos[2], other_z + 0.05, atol=STACKING_APPROACH_DISTANCE):         # FIXME: 0.05 is the cube size, avoid hardcoding
                        stacking_pairs.add((other_obj_id, object_id))  # (Bottom, Top)
                    elif np.isclose(other_z, pos[2] + 0.05, atol=STACKING_APPROACH_DISTANCE):
                        stacking_pairs.add((object_id, other_obj_id))  # (Bottom, Top)

        stacking_pairs = list(stacking_pairs)

        return True, stacking_pairs


    def _object_is_correctly_stacked(self, object_id, orbit):
        """Return whether this object belongs to the completed stack prefix."""
        attached_object_ids = {
            attached_object_id
            for attached_object_id in orbit.attachments.values()
            if attached_object_id is not None
        }

        # Object IDs define their bottom-to-top order. An upper object is only
        # complete when it and every supporting object are placed at their
        # respective goals and none of them is currently being carried.
        for stack_object_id in range(object_id + 1):
            if stack_object_id in attached_object_ids:
                return False

            placement_pose = orbit.placement_poses.get(stack_object_id)
            if placement_pose is None:
                return False

            if not np.allclose(
                np.asarray(placement_pose, dtype=float)[:3, 3],
                self.mobjects[stack_object_id].get_goal_position(),
                atol=0.02,
            ):
                return False

        return True

    def get_object_goal_status(self, orbit):
        """Report completion of the valid bottom-to-top stack prefix."""
        return tuple(
            self._object_is_correctly_stacked(obj.id, orbit)
            for obj in sorted(self.mobjects, key=lambda item: item.id)
        )

    def get_modal_state_cost(self, modal_state):
        """Score stack progress while strongly penalizing bad precedence.

        This is a high-level sampling heuristic, not a feasibility test.
        Lower values are preferred. A useful look-ahead grasp is allowed, but
        out-of-order progress, far-ahead grasps, and blocking both robot hands
        before acquiring the next required object are discouraged.
        """
        ordered_object_ids = [
            obj.id for obj in sorted(self.mobjects, key=lambda item: item.id)
        ]
        progress = tuple(bool(value) for value in modal_state.at_goal)
        held_objects = {
            object_id
            for object_id in modal_state.mode.attachments
            if object_id is not None
        }

        # Generic admissible-style pick/place estimate: an unfinished object
        # needs one action if already held and two otherwise.
        cost = sum(
            0 if complete else (1 if object_id in held_objects else 2)
            for object_id, complete in zip(ordered_object_ids, progress)
        )

        first_incomplete = next(
            (
                index
                for index, complete in enumerate(progress)
                if not complete
            ),
            len(progress),
        )

        # A True entry above a False support represents out-of-order task
        # progress (for example object 2 placed before object 1).
        precedence_violations = sum(
            1
            for index, complete in enumerate(progress)
            if complete and not all(progress[:index])
        )
        cost += 20 * precedence_violations

        # Preparing at most one object per robot near the active stack level
        # is useful. Holding objects farther ahead consumes scarce hands.
        useful_frontier = set(
            ordered_object_ids[
                first_incomplete:first_incomplete + self.num_robots
            ]
        )
        far_ahead_held = held_objects - useful_frontier
        cost += 5 * len(far_ahead_held)

        if first_incomplete < len(ordered_object_ids):
            required_object = ordered_object_ids[first_incomplete]
            all_hands_busy = all(
                object_id is not None
                for object_id in modal_state.mode.attachments
            )
            if all_hands_busy and required_object not in held_objects:
                cost += 50

        return float(cost)

    def get_modal_pick_candidates(
        self,
        modal_state,
        robot_id,
        available_objects,
    ):
        """Keep symbolic guidance near the next useful stack frontier.

        This only ranks high-level actions. The boundary sampler and collision
        checker still decide whether a robot can actually perform the pick.
        Up to one object per robot is exposed so parallel preparation remains
        possible without guiding the search toward arbitrary upper layers.
        """
        del robot_id
        first_incomplete = next(
            (
                index
                for index, is_complete in enumerate(modal_state.at_goal)
                if not is_complete
            ),
            len(modal_state.at_goal),
        )
        frontier_ids = {
            obj.id
            for obj in sorted(self.mobjects, key=lambda item: item.id)[
                first_incomplete:first_incomplete + self.num_robots
            ]
        }
        return [
            object_id
            for object_id in available_objects
            if object_id in frontier_ids
        ]

    def is_modal_place_allowed(self, modal_state, robot_id, object_id):
        """Guide placement only to the next unfinished stack level."""
        del robot_id
        first_incomplete = next(
            (
                index
                for index, is_complete in enumerate(modal_state.at_goal)
                if not is_complete
            ),
            None,
        )
        if first_incomplete is None:
            return False
        ordered_objects = sorted(self.mobjects, key=lambda item: item.id)
        return object_id == ordered_objects[first_incomplete].id

    def _pickable_object_ids(self, orbit):
        """Return unattached objects outside the completed stack prefix."""
        attached_object_ids = {
            object_id
            for object_id in orbit.attachments.values()
            if object_id is not None
        }
        return [
            obj.id
            for obj in self.mobjects
            if obj.id not in attached_object_ids
            and not self._object_is_correctly_stacked(obj.id, orbit)
        ]


    def sample_boundary_configuration(self, orbit):
        '''
            Sample single configuration that lies on the boundary of this orbit.

            joint_angles: list of joint angles for all robots
                            joint angles is sampled based on IK for grasp and place, and random sample for robot not performing transition.

            pre_pose: is the safety pose that the robot need to reach before and after the grasp/ place pose.
                        pre_pose is computed based on the grasp and place pose, with a small offset along the approach direction,
                        or same with the random joint_angles if the robot is not performing transition action.
            
            new_attachments: new attachments after the transition action, should be different with the current attachments in the orbit.
            
            new_grasp_poses: only for the robot that is performing pick action, the new grasp pose for the object that is being picked.
            
            new_placement_poses: only for the robot that is performing place action, the new placement pose for the object that is being placed.
        '''


        # choose the subset of robots \Re
        transition_robots_id = random.choice(self.transition_robot_subsets)

        # available objects for pick action
        available_objects_id = self._pickable_object_ids(orbit)

        # Sample configuration for all robots
        joint_angles = []
        pre_pose = []
        new_attachments = copy.deepcopy(orbit.attachments)
        new_grasp_poses = {}
        new_placement_poses = {}
        for r_id in self.robot_list:
            if r_id in transition_robots_id:   # this robot is doing transition action
                # --------------------------
                # PICK ACTION
                # --------------------------
                if orbit.attachments[r_id] is None:  
                    if len(available_objects_id) == 0:
                        # No available objects to pick, this robot cannot do pick action, skip pick but random a joint angle
                        rand_joint_angles = self.random_joint_angles_for_robot(r_id)
                        joint_angles.append(rand_joint_angles)
                        pre_pose.append(rand_joint_angles)
                        continue
                    else:
                        obj_id = random.choice(available_objects_id)
                        r_joint_angles, r_joint_angle_pre, T_grasp = sample_grasp_pose(obj_id, self, r_id, orbit)
                        if T_grasp is None or r_joint_angles is None:
                            # robot cannot find feasible grasp pose for this object, skip pick but random a joint angle
                            rand_joint_angles = self.random_joint_angles_for_robot(r_id)
                            joint_angles.append(rand_joint_angles)
                            pre_pose.append(rand_joint_angles)  # add the random joint angles as the pre_pose
                            continue
                        r_joint_angles = np.round(r_joint_angles, 4).tolist()
                        r_joint_angle_pre = np.round(r_joint_angle_pre, 4).tolist()
                        # update
                        joint_angles.append(r_joint_angles)
                        pre_pose.append(r_joint_angle_pre)
                        new_grasp_poses[obj_id] = T_grasp
                        new_attachments[r_id] = obj_id
                        available_objects_id.remove(obj_id)
                else:   
                    # --------------------------
                    # PLACE ACTION
                    # --------------------------
                    obj_id = orbit.attachments[r_id]
                    x_place = self.mobjects[obj_id].x_goal  # FIXME: this is ideal case
                    y_place = self.mobjects[obj_id].y_goal
                    z_place = self.mobjects[obj_id].z_goal
                    r_joint_angles, r_joint_angles_pre, T_object = sample_placement_pose(obj_id, orbit.grasp_poses[obj_id], 
                                                                        x_goal=x_place, 
                                                                        y_goal=y_place, 
                                                                        z_goal=z_place,
                                                                        environment=self, r_id=r_id, orbit=orbit)
                    if T_object is None or r_joint_angles is None:
                        # skip place but random a joint angle bcs robot can keep holding
                        rand_joint_angles = self.random_joint_angles_for_robot(r_id)
                        joint_angles.append(rand_joint_angles)
                        pre_pose.append(rand_joint_angles)  # add the random joint angles as the pre_pose
                        continue
                    else:
                        r_joint_angles = np.round(r_joint_angles, 4).tolist()
                        r_joint_angles_pre = np.round(r_joint_angles_pre, 4).tolist()
                        # update
                        joint_angles.append(r_joint_angles)
                        pre_pose.append(r_joint_angles_pre)
                        new_placement_poses[obj_id] = T_object
                        new_attachments[r_id] = None
            else:
                # this robot is not doing transition action, just sample random joint angles
                rand_joint_angles = self.random_joint_angles_for_robot(r_id)
                joint_angles.append(rand_joint_angles)
                pre_pose.append(rand_joint_angles)  # add the random joint angles as the pre_pose

        # Check if this is feasible
        if len(new_grasp_poses) + len(new_placement_poses) == 0:
            return None   # no actual transition action sampled, skip

        if len(pre_pose) != self.num_robots or len(joint_angles) != self.num_robots:
            raise ValueError("Mismatch in number of robots and sampled joint angles or pre poses.")


        sampled_configuration = Configuration(
            joint_angles=joint_angles,
            pre_pose=pre_pose,
            attachments=new_attachments,
            grasp_poses=new_grasp_poses,
            placement_poses=new_placement_poses,
            is_transition=True,
            parent_orbit=orbit
        )
        if not self.is_feasible_at_boundary(sampled_configuration, orbit):
            return None
        return sampled_configuration



    def is_feasible_at_boundary(self, configuration, orbit):
        """
        Check if the given configuration is feasible at the boundary of the given orbit (considering attachments, grasp/placement poses)
        """

        # Set robot to the test configuration
        t_ee_poses = {}
        for r_id in range(self.num_robots):
            q_sample = configuration.joint_angles[r_id]
            if self.is_self_colliding(r_id, q_sample):
                return False
            # is_self_colliding already reset the robot to q_sample.
            # Now set the gripper state based on attachment
            gripper_target = 0.0 if configuration.attachments.get(r_id) is not None else 0.04
            p.resetJointState(self.robots[r_id].sim_id, 9, gripper_target, physicsClientId=self.using_client)
            p.resetJointState(self.robots[r_id].sim_id, 10, gripper_target, physicsClientId=self.using_client)
            t_ee_poses[r_id] = self.robots[r_id].get_end_effector_pose()

        # first place all the object in the orbit placement poses and check if they are stable
        is_stable, stacking_pairs = self.stacking_action_check(configuration, orbit)
        if not is_stable:
            return False

        # Object related information
        object_states = {}
        all_object_ids = set()
        if orbit:
            all_object_ids.update(orbit.grasp_poses.keys())
            all_object_ids.update(orbit.placement_poses.keys())
        for object_id in all_object_ids:
            is_attached = object_id in orbit.attachments.values()
            if object_id in configuration.placement_poses:
                obj_status = 'placed'
                obj_pose = configuration.placement_poses[object_id]
            elif object_id in configuration.grasp_poses:
                obj_status = 'grasped'
                obj_pose = configuration.grasp_poses[object_id]
            else:
                obj_status = 'grasped' if is_attached else 'placed'
                obj_pose = orbit.grasp_poses.get(object_id)
                if obj_pose is None:
                    obj_pose = orbit.placement_poses.get(object_id)
                if obj_pose is None:
                    raise ValueError(f"Object{object_id} has no defined pose in the orbit, which is wrong")
            object_states[object_id] = {
                'status': obj_status,
                'pose': obj_pose,
            }

        attachment_sim_ids = {}
        placed_objects_sim_ids = []
        # set object poses in the simulator
        for object_id, state in object_states.items():
            if state['status'] == 'grasped':
                handing_robot_id = None
                holding_robot_id = None
                for r_id, holding_object in configuration.attachments.items():
                    if holding_object == object_id:
                        holding_robot_id = r_id
                        if object_id in orbit.attachments.values():
                            handing_robot_id = [r_id for r_id, obj_id in orbit.attachments.items() 
                                                if obj_id == object_id][0]
                        break
                if holding_robot_id is not None:
                    t_object = t_ee_poses[holding_robot_id] @ state['pose'] 
                    self.mobjects[object_id].set_pose(t_object)
                    attachment_sim_ids[self.robots[holding_robot_id].sim_id] = self.mobjects[object_id].sim_id
                # when grasping, at the boundary point the object still touching the table
                placed_objects_sim_ids.append(self.mobjects[object_id].sim_id)
                # at the handover one object can be held by two robots at the same time
                if handing_robot_id is not None and handing_robot_id != holding_robot_id:
                    attachment_sim_ids[self.robots[handing_robot_id].sim_id] = self.mobjects[object_id].sim_id

            else:
                self.mobjects[object_id].set_pose(state['pose'])
                placed_objects_sim_ids.append(self.mobjects[object_id].sim_id)
                # when placing, at the boundary point the object still touch the gripper
                robots_holding_object = [r_id for r_id, holding_object in orbit.attachments.items() if holding_object == object_id]
                for r_id in robots_holding_object:
                    attachment_sim_ids[self.robots[r_id].sim_id] = self.mobjects[object_id].sim_id

        all_robot_sim_ids = [r.sim_id for r in self.robots]
        if self._has_forbidden_collision(
            all_robot_sim_ids,
            attachment_sim_ids,
            placed_objects_sim_ids,
            stacking_pairs,
            debug_context="boundary",
        ):
            return False
        return True
    

    def is_placement_collision_free(self, object_id, robot_id, robot_joint_angles, orbit):
        """
            Check if the given placement is valid (no collisions)
        
            This placement collision free only checks placment pose of objects is collision free,
            other collision would computed later in the transition configuration feasibility check.
        
        """

        # Robot self collision check
        if self.is_self_colliding(robot_id, robot_joint_angles):
            if print_debug:
                print(f"Robot {robot_id} self-collision detected at joint angles {robot_joint_angles}")
            return False

        # Extract the values
        t_ee = self.robots[robot_id].forward_kinematics(robot_joint_angles)
        current_object_sim_id = self.mobjects[object_id].sim_id
        placed_objects_sim_ids = {}
        for obj_id, placement_pose in orbit.placement_poses.items():
            placed_objects_sim_ids[obj_id] = self.mobjects[obj_id].sim_id

        # Attach this object to robot EE
        object_placement_position = []                              # just need [x,y,z]
        holding_object = orbit.attachments[robot_id]
        if holding_object is not None:
            t_object = (
                t_ee
                @ orbit.grasp_poses[holding_object]
            )
            self.mobjects[holding_object].set_pose(                 # set placement action for this object
                t_object
            )
            object_placement_position = t_object[:3, 3]
            object_placement_position = np.round(object_placement_position, 2).tolist()


        # Place all placed objects and get the stacking pairs
        stacking_pairs = []
        for obj_id, placement_pose in orbit.placement_poses.items():
            self.mobjects[obj_id].set_pose(
                placement_pose
            )
            if obj_id != object_id:
                placement_position = placement_pose[:3, 3]
                placement_position = np.round(placement_position, 2).tolist()
                if object_placement_position and \
                    placement_position[0] == object_placement_position[0] and \
                    placement_position[1] == object_placement_position[1] and \
                    np.isclose(object_placement_position[2], placement_position[2] + 0.05, atol=STACKING_APPROACH_DISTANCE):
                    # This is a stacking pair, allow contact
                    stacking_pairs.append((obj_id, object_id))  # (Bottom, Top)

        
        # Collision checking between placing objs and other movable objects
        for obj_id, obj_sim_id in placed_objects_sim_ids.items():
            if obj_id == object_id:
                continue
            if obj_id in [pair[0] for pair in stacking_pairs] and object_id in [pair[1] for pair in stacking_pairs] \
                or object_id in [pair[0] for pair in stacking_pairs] and obj_id in [pair[1] for pair in stacking_pairs]:
                # This is a stacking pair, allow contact
                continue
            contacts = p.getClosestPoints(obj_sim_id, current_object_sim_id, distance=SAFE_APPROACH_DISTANCE, physicsClientId=self.using_client)
            for c in contacts:
                if c[8] >= SAFE_APPROACH_DISTANCE:
                    continue
                if print_debug:
                    print(f"Collision detected between placing object {object_id} and placed object {obj_id} at contact distance {c[8]}")
                return False
        # Collision checking between placing objects and fixed environment objects
        for fixed_sim_id in self.fixed_objects:
            contacts = p.getClosestPoints(current_object_sim_id, fixed_sim_id, distance=SAFE_APPROACH_DISTANCE, physicsClientId=self.using_client)
            for c in contacts:
                if c[8] >= SAFE_APPROACH_DISTANCE:
                    continue
                if print_debug:
                    print(f"Collision detected between placing object {object_id} and fixed object {fixed_sim_id} at contact distance {c[8]}")
                return False
        
        return True


    ####################################################################
    #======================= Visualization Utils =======================
    ####################################################################
    def _draw_object_targets(self, client_id=None):
        """Visualize target object locations"""
        if client_id is None:
            client_id = self.visualization_client

        color  = [0, 0, 0]  # black

        goal = self.mobjects[0].get_goal_position()

        if isinstance(goal[0], list):
            # Draw a bounding box for the goal region
            min_x = min([g[0] for g in goal])
            max_x = max([g[0] for g in goal])
            min_y = min([g[1] for g in goal])
            max_y = max([g[1] for g in goal])
            min_z = min([g[2] for g in goal])
            max_z = max([g[2] for g in goal])
            p.addUserDebugLine(
                lineFromXYZ=[min_x, min_y, min_z],
                lineToXYZ=[max_x, min_y, min_z],
                lineColorRGB=color[:3],
                lineWidth=2,
                physicsClientId=client_id
            )
        else:
            p.addUserDebugPoints(
                pointPositions=[[goal[0], goal[1], goal[2]]],
                pointColorsRGB=[color],
                pointSize=12,
                physicsClientId=client_id
            )




if __name__ == "__main__":
    env = StackingObjectsEnv(num_robots=2, num_mobjects=4)
    # env.simulate_goal()
    try:
        env.keep_gui_alive()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if sys.gettrace() is None:
            env.close()
        else:
            print("Debugger mode: leaving clients connected on exit.")
