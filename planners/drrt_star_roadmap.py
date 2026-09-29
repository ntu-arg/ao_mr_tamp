

import heapq
import numpy as np
from scipy.spatial import cKDTree

from planners.prm_roadmap_utils import Orbit


class Node:

    """
        A class to represent a node in the constituent roadmap graph.
        Each node contains a configuration in the robot's configuration space, and it's parent orbit. 
    """

    def __init__(
        self,
        joint_angles,
        *parent_orbits,
        is_transition=False,
        transition_key=None,
        attachment=None,
        grasp_pose=None,
        placement_pose=None,
        pre_pose=None,
        is_goal=False,
    ):
        if not parent_orbits:
            raise ValueError("A node must belong to at least one orbit.")

        # This one have same information as composite configuration
        self.joint_angles = joint_angles    # robot joint_angles
        self.attachment = attachment         # this robot holding which object (object id)
        self.grasp_pose = grasp_pose         # if grasping, save the grasp pose here
        self.placement_pose = placement_pose # if placing, save the placement pose here
        self.pre_pose = pre_pose             # if pregrasping, save the pregrasp pose here
        self.is_goal = bool(is_goal)

        self.heuristic_value = None
        # Assigned by ConstituentRoadmap in deterministic insertion order.
        # This is deliberately not a process-global counter: every roadmap
        # starts from zero again for every planner run.
        self.creation_index = None
        self.is_transition = bool(is_transition)
        self._parent_orbits = {id(orbit): orbit for orbit in parent_orbits}
        self._joint_angles_key = _to_hashable(joint_angles)
        # Parent membership may grow for transitions, so it cannot be part of
        # their dictionary identity. Use the identity of the shared composite
        # boundary sample: its per-robot constituent nodes must canonicalize
        # consistently, while an equal-valued sample created from another
        # orbit must remain a distinct transition. Using Configuration value
        # equality here previously merged unrelated orbit boundaries.
        self._transition_key = transition_key if self.is_transition else None
        self._transition_identity = (
            id(transition_key)
            if self.is_transition and transition_key is not None
            else None
        )
        self._identity_key = (
            self._joint_angles_key,
            self.is_transition,
            self._transition_identity,
            None if self.is_transition else frozenset(self._parent_orbits),
        )
        # Node identity is immutable after construction.  In particular,
        # transition parent-orbit membership is deliberately excluded above.
        # Cache the hash once because roadmap searches use nodes as dictionary
        # keys millions of times.
        self._hash = hash(self._identity_key)

    @property
    def parent_orbit(self):
        return tuple(self._parent_orbits.values())

    def add_parent_orbit(self, orbit):
        self._parent_orbits.setdefault(id(orbit), orbit)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented

        return self._identity_key == other._identity_key


class ConstituentRoadmap:

    """
        A class to represent the constituent roadmap graph for a single robot.
    
        Each node contains a configuration in the robot's configuration space, and it's parent orbit. 
        Edges represent local paths; lazy mode defers their collision checks.
        The graph is built incrementally by adding interior and boundary nodes sampled from the robot's orbits.
    
    """


    INTERIOR_TO_TRANSITION_NEIGHBORS = 2
    MAX_TRANSITION_TO_INTERIOR_NEIGHBORS = 24

    def __init__(self, robot_id, init_joint_angles, init_orbit, environment=None, lazy_level1=False):
        self.robot_id = robot_id            # the constitutent roadmap is specific to a single robot
        self.graph = {}
        self.environment = environment
        self.lazy_level1 = lazy_level1
        self._edge_status = {}
        self.lazy_collision_checks = 0
        self.lazy_invalid_edges = 0
        self._nodes_by_orbit = {}
        self._canonical_nodes = {}
        self._connected_orbits_by_node = {}
        self._edge_orbits = {}
        self._orbit_index_nodes = {}
        self._orbit_index_trees = {}
        self._orbit_index_sizes = {}
        self._interior_index_nodes = {}
        self._interior_index_trees = {}
        self._interior_index_sizes = {}
        self._transition_index_nodes = {}
        self._transition_index_trees = {}
        self._transition_index_sizes = {}
        self._transition_nodes_by_orbit = {}
        self._next_node_index = 0
        # Both caches are partitioned by orbit so roadmap growth in one mode
        # does not discard reusable search results from every other mode.
        self._target_distance_cache = {}
        self.target_distance_searches = 0
        self.target_distance_cache_hits = 0

        self.root_node = Node(init_joint_angles, init_orbit)
        self.goal_nodes = []
        self._add_new_node(self.root_node)
        self._attach_node_to_orbit(self.root_node, init_orbit)


    def add_interior_node(self, robot_joint_angles, orbit: Orbit, attachment=None):
        """ Add a new interior node to the roadmap graph and connect it to its nearest neighbors in the same orbit. 
            :param robot_joint_angles: The single robot configuration of the new node.
            :param orbit: The orbit to which the new node belongs.
            :param environment: The environment for collision checking.
        """
        new_node = Node(robot_joint_angles, orbit, attachment=attachment)
        canonical, was_added = self._get_or_add_node(new_node)
        if was_added:
            self._attach_node_to_orbit(canonical, orbit)
            self._connect_to_nearest_neighbors(new_node, orbit)
        return canonical


    def add_boundary_node(self, robot_joint_angles, parent_orbits, grasp_pose=None, placement_pose=None, pre_pose=None, transition_key=None):
        """Add or update a boundary node shared by any number of orbits."""
        unique_orbits = tuple({id(orbit): orbit for orbit in parent_orbits}.values())
        if not unique_orbits:
            raise ValueError("A boundary node must have at least one parent orbit.")

        node = Node(
            robot_joint_angles,
            *unique_orbits,
            is_transition=True,
            transition_key=transition_key,
            grasp_pose=grasp_pose,
            placement_pose=placement_pose,
            pre_pose=pre_pose
        )
        canonical, _ = self._get_or_add_node(node)
        for orbit in unique_orbits:
            if id(orbit) in self._connected_orbits_by_node[canonical]:
                continue
            self._attach_node_to_orbit(canonical, orbit)
            self._connect_to_nearest_neighbors(canonical, orbit)
        return canonical


    @staticmethod
    def _k_nearest_from_index(
        node: Node,
        orbit: Orbit,
        k: int,
        index_nodes,
        index_trees,
        index_sizes,
    ):
        """Return nearest nodes from one type-specific per-orbit index."""
        if k <= 0:
            return []

        orbit_key = id(orbit)
        nodes = index_nodes.get(orbit_key, [])
        if not nodes or (len(nodes) == 1 and nodes[0] is node):
            return []

        indexed_size = index_sizes.get(orbit_key, 0)
        if (
            index_trees.get(orbit_key) is None
            or len(nodes) - indexed_size >= 32
        ):
            coordinates = np.vstack([
                _configuration_coordinates(candidate)
                for candidate in nodes
            ])
            index_trees[orbit_key] = cKDTree(coordinates)
            indexed_size = len(nodes)
            index_sizes[orbit_key] = indexed_size

        query = _configuration_coordinates(node)
        candidate_distances = {}
        tree = index_trees[orbit_key]
        # Query one spare candidate because ``node`` itself may belong to this
        # type-specific index. ``nsmallest`` below removes the spare when the
        # query node belongs to the other index.
        query_count = min(k + 1, indexed_size)
        distances, indices = tree.query(query, k=query_count)
        for candidate_distance, candidate_index in zip(
            np.atleast_1d(distances),
            np.atleast_1d(indices),
        ):
            candidate = nodes[int(candidate_index)]
            if candidate is not node:
                candidate_distances[candidate] = float(candidate_distance)

        for candidate in nodes[indexed_size:]:
            if candidate is not node:
                candidate_distances[candidate] = distance(node, candidate)

        return heapq.nsmallest(
            k,
            candidate_distances,
            key=lambda candidate: (
                candidate_distances[candidate],
                candidate.creation_index,
            ),
        )


    def k_nearest_neighbors(self, node: Node, orbit: Orbit, k: int):
        """Return up to ``k`` roadmap nodes of either type in ``orbit``."""
        return self._k_nearest_from_index(
            node,
            orbit,
            k,
            self._orbit_index_nodes,
            self._orbit_index_trees,
            self._orbit_index_sizes,
        )


    def k_nearest_interior_neighbors(self, node: Node, orbit: Orbit, k: int):
        """Return up to ``k`` non-transition neighbors in ``orbit``."""
        return self._k_nearest_from_index(
            node,
            orbit,
            k,
            self._interior_index_nodes,
            self._interior_index_trees,
            self._interior_index_sizes,
        )


    def k_nearest_transition_neighbors(self, node: Node, orbit: Orbit, k: int):
        """Return up to ``k`` transition neighbors in ``orbit``."""
        return self._k_nearest_from_index(
            node,
            orbit,
            k,
            self._transition_index_nodes,
            self._transition_index_trees,
            self._transition_index_sizes,
        )


    def _add_new_node(self, node: Node):
        node.creation_index = self._next_node_index
        self._next_node_index += 1
        self.graph[node] = {}
        self._canonical_nodes[node] = node
        self._connected_orbits_by_node[node] = set()


    def _get_or_add_node(self, node: Node):
        canonical = self._canonical_nodes.get(node)
        if canonical is not None:
            return canonical, False
        self._add_new_node(node)
        return node, True


    def _attach_node_to_orbit(self, node: Node, orbit: Orbit):
        orbit_key = id(orbit)
        node.add_parent_orbit(orbit)
        orbit_nodes = self._nodes_by_orbit.setdefault(orbit_key, set())
        if node in orbit_nodes:
            return
        orbit_nodes.add(node)
        self._invalidate_orbit_search_caches(orbit)
        self._orbit_index_nodes.setdefault(orbit_key, []).append(node)
        self._orbit_index_trees.setdefault(orbit_key, None)
        self._orbit_index_sizes.setdefault(orbit_key, 0)
        if node.is_transition:
            self._transition_index_nodes.setdefault(orbit_key, []).append(node)
            self._transition_index_trees.setdefault(orbit_key, None)
            self._transition_index_sizes.setdefault(orbit_key, 0)
            if node._transition_key is not None:
                self._transition_nodes_by_orbit.setdefault(orbit_key, {})[
                    id(node._transition_key)
                ] = node
        else:
            self._interior_index_nodes.setdefault(orbit_key, []).append(node)
            self._interior_index_trees.setdefault(orbit_key, None)
            self._interior_index_sizes.setdefault(orbit_key, 0)


    def nodes_in_orbit(self, orbit: Orbit):
        """Return the canonical constituent nodes associated with ``orbit``."""
        return tuple(self._orbit_index_nodes.get(id(orbit), ()))


    def _connect_to_nearest_neighbors(self, node: Node, orbit: Orbit):
        """Connect to the mixed nearest-neighbor set used by the baseline."""
        orbit_key = id(orbit)
        connected_orbits = self._connected_orbits_by_node[node]
        if orbit_key in connected_orbits:
            return

        neighbors = self.k_nearest_neighbors(node, orbit, k=orbit.k)
        topology_changed = False
        for neighbor in neighbors:
            topology_changed |= self._try_add_edge(node, neighbor, orbit)
        connected_orbits.add(orbit_key)
        if topology_changed:
            self._invalidate_orbit_search_caches(orbit)


    def _try_add_edge(self, node: Node, neighbor: Node, orbit: Orbit) -> bool:
        """Add one orbit-context edge while preserving lazy validation state."""
        if node is neighbor:
            return False
        orbit_key = id(orbit)
        edge_key = frozenset((node, neighbor))
        valid_orbits = self._edge_orbits.setdefault(edge_key, set())
        context_key = (orbit_key, edge_key)
        if self._edge_status.get(context_key) == "INVALID":
            return False
        if orbit_key in valid_orbits:
            return False
        if (
            not self.lazy_level1
            and self.environment is not None
            and not self.environment.constituent_motion_collision_checking(
                self.robot_id,
                node,
                neighbor,
                orbit,
            )
        ):
            return False

        edge_distance = distance(node, neighbor)
        self.graph[node][neighbor] = edge_distance
        self.graph[neighbor][node] = edge_distance
        valid_orbits.add(orbit_key)
        self._edge_status[context_key] = (
            "UNKNOWN" if self.lazy_level1 else "VALID"
        )
        return True


    def _invalidate_orbit_search_caches(self, orbit: Orbit):
        """Invalidate search data only for the orbit whose topology changed."""
        orbit_key = id(orbit)
        self._target_distance_cache.pop(orbit_key, None)


    def remove_node(self, node: Node):
        if node not in self.graph:
            return
        affected_orbits = tuple(node.parent_orbit)
        # Remove this node from all neighbors
        for neighbor in list(self.graph[node].keys()):
            if neighbor in self.graph:
                self.graph[neighbor].pop(node, None)
        for edge_key in list(self._edge_orbits):
            if node in edge_key:
                del self._edge_orbits[edge_key]
        for orbit in node.parent_orbit:
            orbit_nodes = self._nodes_by_orbit.get(id(orbit))
            if orbit_nodes is not None:
                orbit_nodes.discard(node)
                if not orbit_nodes:
                    del self._nodes_by_orbit[id(orbit)]
            orbit_key = id(orbit)
            for index_nodes, index_trees, index_sizes in (
                (
                    self._orbit_index_nodes,
                    self._orbit_index_trees,
                    self._orbit_index_sizes,
                ),
                (
                    self._interior_index_nodes,
                    self._interior_index_trees,
                    self._interior_index_sizes,
                ),
                (
                    self._transition_index_nodes,
                    self._transition_index_trees,
                    self._transition_index_sizes,
                ),
            ):
                indexed_nodes = index_nodes.get(orbit_key)
                if indexed_nodes is None:
                    continue
                remaining_nodes = [
                    candidate
                    for candidate in indexed_nodes
                    if candidate is not node
                ]
                if remaining_nodes:
                    index_nodes[orbit_key] = remaining_nodes
                    index_trees[orbit_key] = None
                    index_sizes[orbit_key] = 0
                else:
                    index_nodes.pop(orbit_key, None)
                    index_trees.pop(orbit_key, None)
                    index_sizes.pop(orbit_key, None)
            transition_nodes = self._transition_nodes_by_orbit.get(orbit_key)
            if transition_nodes is not None and node._transition_key is not None:
                transition_key = id(node._transition_key)
                if transition_nodes.get(transition_key) is node:
                    del transition_nodes[transition_key]
                if not transition_nodes:
                    del self._transition_nodes_by_orbit[orbit_key]
        for context_key in list(self._edge_status):
            if node in context_key[1]:
                del self._edge_status[context_key]
        for orbit in affected_orbits:
            self._invalidate_orbit_search_caches(orbit)
        # Remove the node itself
        del self.graph[node]
        self._canonical_nodes.pop(node, None)
        self._connected_orbits_by_node.pop(node, None)


    def get_neighbors(self, node: Node):
        return list(self.graph.get(node, {}).keys())


    def neighbors(self, node: Node, orbit: Orbit):
        """Return neighbors available for optimistic exploration in ``orbit``."""
        return [
            neighbor
            for neighbor in self.graph.get(node, {})
            if self.has_edge_in_orbit(node, neighbor, orbit)
        ]


    def has_edge_in_orbit(self, node_a: Node, node_b: Node, orbit: Orbit):
        """Return whether an edge is available in the given orbit (possibly UNKNOWN)."""
        if node_a is node_b:
            return True
        edge_key = frozenset((node_a, node_b))
        return id(orbit) in self._edge_orbits.get(edge_key, ())


    def validate_edge(self, node_a, node_b, orbit):
        """Validate a constituent motion in exactly this orbit context."""
        if not self.has_edge_in_orbit(node_a, node_b, orbit):
            return False
        if node_a is node_b or not self.lazy_level1:
            return True
        edge_key = frozenset((node_a, node_b))
        context_key = (id(orbit), edge_key)
        if self._edge_status.get(context_key) == "VALID":
            return True
        if self.environment is None:
            raise ValueError("Lazy Level-1 validation requires an environment.")
        self.lazy_collision_checks += 1
        valid = self.environment.constituent_motion_collision_checking(
            self.robot_id, node_a, node_b, orbit
        )
        self._edge_status[context_key] = "VALID" if valid else "INVALID"
        if not valid:
            self.lazy_invalid_edges += 1
            contexts = self._edge_orbits[edge_key]
            contexts.discard(id(orbit))
            if not contexts:
                self.graph[node_a].pop(node_b, None)
                self.graph[node_b].pop(node_a, None)
            self._invalidate_orbit_search_caches(orbit)
            self.update_heuristic_values()
        return valid


    def distances_to_target(self, target: Node, orbit: Orbit):
        """Shortest constituent-roadmap distances to ``target`` in one orbit."""
        if target not in self.graph or orbit not in target.parent_orbit:
            return {}

        # ``target`` is already a canonical roadmap node.  Identity avoids an
        # unnecessary node-hash operation on every cache lookup.
        orbit_cache = self._target_distance_cache.setdefault(id(orbit), {})
        target_key = id(target)
        cached = orbit_cache.get(target_key)
        if cached is not None:
            self.target_distance_cache_hits += 1
            return cached

        self.target_distance_searches += 1

        distances = {target: 0.0}
        priority_queue = [(0.0, 0, target)]
        counter = 1

        while priority_queue:
            current_distance, _, node = heapq.heappop(priority_queue)
            if current_distance > distances[node]:
                continue
            for neighbor, edge_cost in self.graph[node].items():
                if not self.has_edge_in_orbit(node, neighbor, orbit):
                    continue
                candidate_distance = current_distance + edge_cost
                if candidate_distance >= distances.get(neighbor, float("inf")):
                    continue
                distances[neighbor] = candidate_distance
                heapq.heappush(
                    priority_queue,
                    (candidate_distance, counter, neighbor),
                )
                counter += 1
        orbit_cache[target_key] = distances
        return distances


    def transition_node(self, transition_key, orbit: Orbit):
        """Return this robot's component of a composite transition."""
        return self._transition_nodes_by_orbit.get(id(orbit), {}).get(
            id(transition_key)
        )

    def transition_sources_in_orbit(self, orbit: Orbit):
        """Return composite transition keys indexed for ``orbit``."""
        return tuple(
            node._transition_key
            for node in self._transition_nodes_by_orbit.get(
                id(orbit), {}
            ).values()
        )

    
    def add_goal_node(self, goal_joint_angles, goal_orbit: Orbit):
        """
            Set the goal node for the constituent roadmap graph.
            :param goal_joint_angles: The single robot configuration of the goal node.
        """
        new_goal_node = Node(
            goal_joint_angles,
            goal_orbit,
            is_transition=False,
            is_goal=True,
        )
        canonical_goal_node = self._canonical_nodes.get(new_goal_node)
        if canonical_goal_node is not None:
            canonical_goal_node.is_goal = True
            if canonical_goal_node not in self.goal_nodes:
                self.goal_nodes.append(canonical_goal_node)
            return canonical_goal_node

        if (
            self.environment is not None
            and not self.environment.constituent_node_collision_checking(
                self.robot_id,
                goal_joint_angles,
                goal_orbit,
            )
        ):
            return None
        canonical_goal_node, was_added = self._get_or_add_node(new_goal_node)
        canonical_goal_node.is_goal = True

        if was_added:
            self._attach_node_to_orbit(canonical_goal_node, goal_orbit)
            self._connect_to_nearest_neighbors(canonical_goal_node, goal_orbit)

        if canonical_goal_node not in self.goal_nodes:
            self.goal_nodes.append(canonical_goal_node)
        return canonical_goal_node


    def update_heuristic_values(self):
        """
        Calculate shortest-path distance to the nearest goal for every node.

        This is Algorithm 7's multi-source Dijkstra: all goal nodes enter the
        priority queue with distance zero, so the roadmap is traversed once.
        """
        for node in self.graph:
            node.heuristic_value = float('inf')

        if not self.goal_nodes:
            return

        priority_queue = []
        counter = 0
        for goal_node in self.goal_nodes:
            if goal_node not in self.graph or goal_node.heuristic_value == 0.0:
                continue
            goal_node.heuristic_value = 0.0
            heapq.heappush(priority_queue, (0.0, counter, goal_node))
            counter += 1

        while priority_queue:
            heuristic, _, node = heapq.heappop(priority_queue)
            if heuristic > node.heuristic_value:
                continue

            for neighbor, edge_cost in self.graph[node].items():
                candidate = heuristic + edge_cost
                if candidate >= neighbor.heuristic_value:
                    continue
                neighbor.heuristic_value = candidate
                heapq.heappush(
                    priority_queue,
                    (candidate, counter, neighbor),
                )
                counter += 1



def distance(node1, node2):
    """ Simply compute the Euclidean distance between two nodes based on their configurations."""

    # Assuming node1 and node2 are instances of Node class
    return np.linalg.norm(
        _configuration_coordinates(node1)
        - _configuration_coordinates(node2)
    )


def _configuration_coordinates(configuration):
    """Return a flat numeric vector for a raw or Configuration-like value."""
    if (
        getattr(configuration, "is_transition", False)
        and getattr(configuration, "pre_pose", None) is not None
    ):
        values = configuration.pre_pose
    else:
        values = getattr(configuration, "joint_angles", configuration)
    return np.asarray(values, dtype=float).reshape(-1)


def _to_hashable(value):
    """Create a stable equality/hash key for common configuration containers."""
    if hasattr(value, "joint_angles"):
        return _to_hashable(value.joint_angles)
    if isinstance(value, np.ndarray):
        return (value.shape, tuple(value.reshape(-1).tolist()))
    if isinstance(value, (list, tuple)):
        return tuple(_to_hashable(item) for item in value)
    if isinstance(value, dict):
        return frozenset(
            (_to_hashable(key), _to_hashable(item))
            for key, item in value.items()
        )
    try:
        hash(value)
    except TypeError:
        raise TypeError(
            f"Unsupported mutable configuration type: {type(value).__name__}"
        )
    return value


def dijkstra(roadmap_graph, start_node, target_node):
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
