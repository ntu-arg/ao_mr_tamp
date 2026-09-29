

import copy
import heapq
import random
import time
import numpy as np

from planners.prm_roadmap_utils import Configuration, Orbit
from planners.drrt_star_roadmap import ConstituentRoadmap


class TensorProductNode:
    def __init__(self, constituent_nodes=None, parent_orbit=None):
        """
        A class to represent a node in the tensor product tree.
        """
        if parent_orbit is None:
            raise ValueError("A TensorProductNode initialization must have a parent orbit.")
        if constituent_nodes is None or not constituent_nodes:
            raise ValueError("A TensorProductNode must have at least one constituent roadmap node.")

        self.constituent_nodes = constituent_nodes if constituent_nodes is not None else []
        self.parent_orbit = parent_orbit if isinstance(parent_orbit, list) else [parent_orbit]
        self.children = []      # list of child nodes in the tree
        self.parent = None      # pointer to the parent node in the tree
        # Orbit in which the current parent edge is executed. This is needed
        # when that edge is validated later as part of a candidate solution.
        self.parent_edge_orbit = None
        self.cost = 0.0         # Cost from root to this node
        self.heuristic_value = None  # Heuristic value for this node
        self.joint_angles = []
        # TensorProductTree assigns this in deterministic insertion order.
        self.creation_index = None

        # Verify if the constituent nodes share the same parent orbit
        for orbit in self.parent_orbit:
            if not all(orbit in node.parent_orbit for node in self.constituent_nodes):
                raise ValueError("All constituent nodes must share the same parent orbit.")
            
        # compute the heuristic value for this node based on the constituent roadmap nodes
        self.heuristic_value = sum(node.heuristic_value for node in self.constituent_nodes if node.heuristic_value is not None)

        # simply store the joint angles of the constituent nodes for easy access
        self.joint_angles = [node.joint_angles for node in self.constituent_nodes]

        self.identity_key = (
            tuple(id(node) for node in self.constituent_nodes),
            frozenset(id(orbit) for orbit in self.parent_orbit),
        )


    def __hash__(self):
        return hash(self.identity_key)

    def __eq__(self, other):
        if not isinstance(other, TensorProductNode):
            return NotImplemented
        return self.identity_key == other.identity_key


    def to_composite_configuration(self) -> Configuration:
        """Convert this internal tree node to the public path format."""
        common_orbits = list(self.constituent_nodes[0].parent_orbit)
        for constituent_node in self.constituent_nodes[1:]:
            common_orbits = [
                orbit
                for orbit in common_orbits
                if orbit in constituent_node.parent_orbit
            ]
        if not common_orbits:
            common_orbits = list(self.parent_orbit)

        transition_sources = {
            id(constituent_node._transition_key): constituent_node._transition_key
            for constituent_node in self.constituent_nodes
            if constituent_node._transition_key is not None
        }
        if len(transition_sources) > 1:
            raise ValueError(
                "A tensor transition node combines incompatible transition samples."
            )

        is_complete_transition = (
            len(transition_sources) == 1
            and all(
                constituent_node.is_transition
                for constituent_node in self.constituent_nodes
            )
        )
        if is_complete_transition:
            source = next(iter(transition_sources.values()))
            return Configuration(
                joint_angles=self.joint_angles,
                attachments=dict(source.attachments),
                is_transition=True,
                parent_orbit=common_orbits,
                grasp_poses=dict(source.grasp_poses),
                placement_poses=dict(source.placement_poses),
                pre_pose=list(source.pre_pose),
            )

        # A transition component is only at its contact configuration when all
        # constituents complete the same transition.  Until then, it waits at
        # its pre-pose while the other robots continue moving.
        joint_angles = [
            constituent_node.pre_pose
            if constituent_node.is_transition
            and constituent_node.pre_pose is not None
            else constituent_node.joint_angles
            for constituent_node in self.constituent_nodes
        ]

        orbit = common_orbits[0]
        return Configuration(
            joint_angles=joint_angles,
            attachments=dict(orbit.attachments),
            is_transition=False,
            parent_orbit=common_orbits,
        )


    
class TensorProductTree:
    EDGE_UNKNOWN = "UNKNOWN"
    EDGE_VALID = "VALID"
    EDGE_INVALID = "INVALID"

    def __init__(self, root_tensor_node, root_orbit, environment):
        self.root = root_tensor_node
        self.orbit_experience = [root_orbit]  # List of orbits that have been explored
        self.V_last = None
        self.Q_target = None
        self.O_target = None
        self.O_next_target = None
        self.S_target = None
        self.guided_steps = 0
        self.max_guided_steps = 25
        self.best_path = None
        self.best_cost = float('inf')
        self.reached_goal_nodes = set()
        self.successful_rewires = 0
        self.root.creation_index = 0
        self._next_node_index = 1

        # All tree nodes: node_key -> TensorProductNode
        self.nodes = {
            self.root.identity_key: self.root
        }

        # Nodes grouped by orbit
        self.orbit_nodes = {
            id(root_orbit): {self.root.identity_key}
        }
        # Parallel insertion-ordered index used whenever traversal order can
        # affect a seeded decision. The set above remains for fast membership
        # and intersection operations.
        self._orbit_node_order = {
            id(root_orbit): [self.root.identity_key]
        }

        self.environment = environment
        self.constituent_roadmaps = None
        # Level-2 tensor edges are optimistic until they occur on a candidate
        # goal path. Collision validity is symmetric, while motion cost can be
        # directional because entering a transition adds its action cost.
        self._tensor_edge_status = {}
        self.lazy_edge_validation_checks = 0
        self.lazy_invalid_edges = 0


    def expand_tree(
        self,
        constituent_roadmaps,
        max_iterations,
        preferred_orbit=None,
        preferred_next_orbit=None,
        desired_modal_state=None,
        modal_graph=None,
        modal_transition_bias=0.8,
        deadline=None,
    ):

        """
        Expand the tree by adding a new node and connecting it to its nearest neighbors in the same orbit.
        
        :param constituent_roadmaps: A dictionary of constituent roadmaps for each robot.
        :param heuristic_values: A dictionary of heuristic values for roadmap nodes of each robot.
        :param max_iterations: The maximum number of iterations to perform.
        :param preferred_orbit: Tensor-experienced orbit selected as the focus
            of this expansion batch.
        :param preferred_next_orbit: Exact successor on an ordered incumbent
            orbit path. It biases transition-target selection only.
        :param desired_modal_state: Preferred symbolic successor of
            ``preferred_orbit``. It only biases transition selection.
        :param modal_graph: Modal graph used to project transition targets.
        :param modal_transition_bias: Probability of restricting a transition
            draw to candidates matching ``desired_modal_state`` when matches
            exist. The complementary probability preserves exploration.
        :param deadline: Optional ``time.perf_counter()`` deadline. It is a
            soft limit because an in-progress collision/IK call is not
            interrupted.
        """

        self.constituent_roadmaps = constituent_roadmaps
        for node in self.nodes.values():
            if self.goal_reached(node):
                self.reached_goal_nodes.add(node)
        preferred_orbit_is_reachable = any(
            orbit is preferred_orbit
            for orbit in self.orbit_experience
        )

        for _ in range(max_iterations):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if self.V_last is not None:
                O_selection = self.O_target
                V_near = self.V_last
                Q_rand = self.Q_target
                active_modal_target = self.S_target
                active_next_orbit = self.O_next_target
            else:
                if preferred_orbit_is_reachable:
                    O_selection = preferred_orbit
                else:
                    live_orbits = [
                        orbit
                        for orbit in self.orbit_experience
                        if self.orbit_nodes.get(id(orbit))
                    ]
                    if not live_orbits:
                        continue
                    O_selection = random.choice(live_orbits)
                active_modal_target = (
                    desired_modal_state
                    if O_selection is preferred_orbit
                    else None
                )
                active_next_orbit = (
                    preferred_next_orbit
                    if O_selection is preferred_orbit
                    else None
                )
                Q_rand = None
                if O_selection.is_goal_orbit:
                    Q_rand = self.sample_goal_configuration(
                        O_selection,
                        constituent_roadmaps,
                    )
                elif random.random() < 0.8:     # FIXME: hardcoded
                    Q_rand = self.sample_reachable_transition(
                        None,
                        O_selection,
                        constituent_roadmaps,
                        desired_modal_state=active_modal_target,
                        desired_next_orbit=active_next_orbit,
                        modal_graph=modal_graph,
                        modal_transition_bias=modal_transition_bias,
                    )
                if Q_rand is None:
                    Q_rand = self.sample_roadmap_configuration(
                        O_selection,
                        constituent_roadmaps,
                    )
                if Q_rand is None:
                    continue
                V_near = self.find_nearest_node(Q_rand, O_selection)
                if Q_rand.is_transition:
                    V_near = self.find_nearest_reachable_transition_node(
                        Q_rand,
                        O_selection,
                        constituent_roadmaps,
                        nearest_node=V_near,
                    )
                if V_near is None:
                    continue
                self.Q_target = Q_rand
                self.O_target = O_selection
                self.O_next_target = active_next_orbit
                self.S_target = active_modal_target
                self.guided_steps = 0

            V_new = informed_extend(V_near, Q_rand, O_selection, constituent_roadmaps, self.environment)
            if V_new is None:
                self.clear_guided_branch()
                continue

            neighbors = self.get_neighbors(
                V_new,
                O_selection,
                constituent_roadmaps,
            )
            if (
                V_near.identity_key != V_new.identity_key
                and self.is_implicit_tensor_edge(
                    V_near,
                    V_new,
                    O_selection,
                    constituent_roadmaps,
                )
                and V_near not in neighbors
            ):
                neighbors.append(V_near)

            V_best = self.choose_best_parent(V_new, neighbors, O_selection)

            if V_best is None:
                self.clear_guided_branch()
                continue

            existing_node = self.nodes.get(V_new.identity_key)
            if existing_node is None:
                self.add_node(V_new, parent_node=V_best, parent_orbit=O_selection)
            else:
                V_new = existing_node
                candidate_cost = V_best.cost + self.compute_cost(
                    V_best, V_new, O_selection
                )
                if candidate_cost < V_new.cost:
                    self.rewire_node(
                        node=V_new,
                        new_parent=V_best,
                        orbit=O_selection,
                    )

            for neighbor in neighbors:
                if neighbor is not V_new and neighbor != V_best:
                    estimated_cost = V_new.cost + self.estimate_edge_cost(
                        V_new,
                        neighbor,
                    )
                    if estimated_cost < neighbor.cost:
                        self.rewire_node(
                            node=neighbor,
                            new_parent=V_new,
                            orbit=O_selection,
                        )

            for orbit in V_new.parent_orbit:
                if orbit.is_goal_orbit:
                    self.try_connect_goal(
                        V_new,
                        orbit,
                        constituent_roadmaps,
                    )

            reached_target = self.target_reached(V_new, Q_rand)
            if Q_rand.is_transition:
                made_progress = (
                    self.transition_target_distance(
                        V_new,
                        Q_rand,
                        O_selection,
                        constituent_roadmaps,
                    )
                    < self.transition_target_distance(
                        V_near,
                        Q_rand,
                        O_selection,
                        constituent_roadmaps,
                    ) - 1e-12
                )
            else:
                made_progress = (
                    self.target_distance(V_new, Q_rand)
                    < self.target_distance(V_near, Q_rand) - 1e-12
                )
            completed_transition = self.is_transition_configuration(V_new)

            if completed_transition:
                next_orbits = self.get_next_orbit(V_new, O_selection)
                if active_next_orbit is not None or (
                    active_modal_target is not None
                    and modal_graph is not None
                ):
                    next_orbits.sort(
                        key=lambda orbit: (
                            active_next_orbit is not None
                            and orbit is not active_next_orbit,
                            active_modal_target is not None
                            and modal_graph is not None
                            and not modal_graph.advances_toward_modal_state(
                                O_selection.parent_modal_state,
                                modal_graph.modal_state_for_orbit(orbit),
                                active_modal_target,
                            ),
                        )
                    )
                for next_orbit in next_orbits:
                    if next_orbit not in self.orbit_experience:
                        self.orbit_experience.append(next_orbit)
                self.clear_guided_branch()
            elif reached_target:
                self.clear_guided_branch()
            elif made_progress and self.guided_steps + 1 < self.max_guided_steps:
                self.V_last = V_new
                self.guided_steps += 1
            else:
                self.clear_guided_branch()

            if self.goal_reached(V_new):
                self.reached_goal_nodes.add(V_new)

        # Keep all inner expansions optimistic and perform lazy solution-edge
        # validation only once after the complete expansion batch.
        self._refresh_best_solution()
        return self.best_path, self.best_cost


    def clear_guided_branch(self):
        """Release the current propagation target and resume exploration."""
        self.V_last = None
        self.Q_target = None
        self.O_target = None
        self.O_next_target = None
        self.S_target = None
        self.guided_steps = 0


    def target_distance(
        self,
        node: TensorProductNode,
        target: Configuration,
    ) -> float:
        return self.compute_distance(node.joint_angles, target.joint_angles)


    @staticmethod
    def transition_target_distance(
        node: TensorProductNode,
        target: Configuration,
        orbit: Orbit,
        constituent_roadmaps: dict[int, ConstituentRoadmap],
    ) -> float:
        total_distance = 0.0
        for roadmap, current_node in zip(
            constituent_roadmaps.values(),
            node.constituent_nodes,
        ):
            target_node = roadmap.transition_node(target, orbit)
            if target_node is None:
                return float("inf")
            distances = roadmap.distances_to_target(target_node, orbit)
            distance = distances.get(current_node, float("inf"))
            if not np.isfinite(distance):
                return float("inf")
            total_distance += distance
        return total_distance


    @staticmethod
    def target_reached(
        node: TensorProductNode,
        target: Configuration,
    ) -> bool:
        if target.is_transition:
            return (
                all(component.is_transition for component in node.constituent_nodes)
                and all(
                    component._transition_key is target
                    for component in node.constituent_nodes
                )
            )
        return all(
            np.allclose(component.joint_angles, target_angles)
            for component, target_angles in zip(
                node.constituent_nodes,
                target.joint_angles,
            )
        )


    def _refresh_best_solution(self):
        """Validate optimistic goal paths and retain the best executable one."""
        while True:
            candidates = []
            for goal_node in sorted(
                self.reached_goal_nodes,
                key=lambda node: node.creation_index,
            ):
                if (
                    self.nodes.get(goal_node.identity_key) is not goal_node
                    or not np.isfinite(goal_node.cost)
                ):
                    continue
                candidate_path = self.retrace_path(goal_node)
                if not candidate_path or candidate_path[0] is not self.root:
                    continue
                _, candidate_cost = self.to_composite_path(candidate_path)
                if candidate_cost < self.best_cost:
                    candidates.append((candidate_cost, goal_node, candidate_path))

            if not candidates:
                return

            candidates.sort(
                key=lambda candidate: (
                    candidate[0],
                    candidate[1].creation_index,
                )
            )
            candidate_cost, _, candidate_path = candidates[0]
            if not self._validate_constituent_path(candidate_path):
                self._remove_invalid_constituent_branches()
                continue
            invalid_edge = self._first_invalid_edge(candidate_path)
            if invalid_edge is not None:
                _, invalid_child, _ = invalid_edge
                self._remove_subtree(invalid_child)
                # Removing the failed branch may expose another optimistic
                # goal candidate. Continue until one validates or none remain.
                continue

            self.best_path = list(candidate_path)
            self.best_cost = candidate_cost
            return


    def _validate_constituent_motion(self, parent, child, orbit):
        if self.constituent_roadmaps is None:
            return True
        for roadmap, start, goal in zip(
            self.constituent_roadmaps.values(),
            parent.constituent_nodes, child.constituent_nodes,
        ):
            if getattr(roadmap, "lazy_level1", False):
                if not roadmap.validate_edge(start, goal, orbit):
                    return False
        return True


    def _validate_constituent_path(self, path):
        """Finish the whole candidate's Level 1 before any Level-2 checks."""
        for parent, child in zip(path, path[1:]):
            if not self._validate_constituent_motion(
                parent, child, child.parent_edge_orbit
            ):
                return False
        return True


    def _remove_invalid_constituent_branches(self):
        """Discard all tree branches using newly unavailable Level-1 edges."""
        for node in tuple(self.nodes.values()):
            if self.nodes.get(node.identity_key) is not node or node.parent is None:
                continue
            if not self.is_implicit_tensor_edge(
                node.parent, node, node.parent_edge_orbit, self.constituent_roadmaps
            ):
                self._remove_subtree(node)
        for node in self.nodes.values():
            node.heuristic_value = self.compute_composite_heuristic(node)


    def _first_invalid_edge(self, path):
        """Validate UNKNOWN edges in a root-to-goal path in execution order."""
        for parent, child in zip(path, path[1:]):
            if child.parent is not parent:
                return parent, child, child.parent_edge_orbit
            orbit = child.parent_edge_orbit
            if orbit is None:
                raise RuntimeError("A non-root tensor node is missing its parent edge orbit.")
            if not self.validate_tensor_edge(parent, child, orbit):
                return parent, child, orbit
        return None


    def _remove_subtree(self, subtree_root):
        """Remove a branch whose root is connected through an invalid edge."""
        if subtree_root is self.root:
            raise RuntimeError("The tensor-tree root cannot be removed.")

        removed_nodes = []
        stack = [subtree_root]
        visited = set()
        while stack:
            node = stack.pop()
            if node.identity_key in visited:
                continue
            visited.add(node.identity_key)
            removed_nodes.append(node)
            stack.extend(node.children)

        removed_set = set(removed_nodes)
        if subtree_root.parent is not None:
            subtree_root.parent.children = [
                child
                for child in subtree_root.parent.children
                if child is not subtree_root
            ]

        if self.V_last in removed_set:
            self.clear_guided_branch()

        self.reached_goal_nodes.difference_update(removed_set)
        for node in removed_nodes:
            if self.nodes.get(node.identity_key) is node:
                del self.nodes[node.identity_key]
            for orbit in node.parent_orbit:
                orbit_key = id(orbit)
                orbit_node_keys = self.orbit_nodes.get(orbit_key)
                if orbit_node_keys is None:
                    continue
                orbit_node_keys.discard(node.identity_key)
                if not orbit_node_keys:
                    del self.orbit_nodes[orbit_key]
                ordered_keys = self._orbit_node_order.get(orbit_key)
                if ordered_keys is not None:
                    self._orbit_node_order[orbit_key] = [
                        key
                        for key in ordered_keys
                        if key != node.identity_key
                    ]
                    if not self._orbit_node_order[orbit_key]:
                        del self._orbit_node_order[orbit_key]

        for node in removed_nodes:
            node.parent = None
            node.parent_edge_orbit = None
            node.children = []


    @staticmethod
    def sample_reachable_transition(
        _near_node: TensorProductNode | None,
        orbit: Orbit,
        constituent_roadmaps: dict[int, ConstituentRoadmap],
        desired_modal_state=None,
        desired_next_orbit=None,
        modal_graph=None,
        modal_transition_bias=0.8,
    ) -> Configuration | None:
        """Randomly sample a shared transition, with optional modal bias.

        This function selects the continuous transition target only. Its
        roadmap reachability is checked later by ``informed_extend``. Avoiding
        shortest-path searches for every available transition prevents the
        accumulated boundary samples from multiplying graph-search work.
        """
        shared_sources = None
        for con_roadmap in constituent_roadmaps.values():
            transition_source_lookup = getattr(
                con_roadmap,
                "transition_sources_in_orbit",
                None,
            )
            if transition_source_lookup is None:
                transition_sources = (
                    candidate._transition_key
                    for candidate in con_roadmap.nodes_in_orbit(orbit)
                    if candidate.is_transition
                    and candidate._transition_key is not None
                )
            else:
                transition_sources = transition_source_lookup(orbit)
            sources = {
                id(source): source
                for source in transition_sources
            }
            if shared_sources is None:
                shared_sources = sources
            else:
                # Filter the first roadmap's insertion-ordered candidates in
                # place. A set intersection here made random.choice map the
                # same seeded index to different transitions across runs.
                shared_sources = {
                    key: source
                    for key, source in shared_sources.items()
                    if key in sources
                }
            if not shared_sources:
                return None

        candidate_sources = list(shared_sources.values())
        guided_sources = []

        # Exact incumbent-edge guidance has priority over the weaker modal
        # target. A source matches only when every constituent transition node
        # belongs to the requested successor orbit.
        if desired_next_orbit is not None:
            guided_sources = [
                source
                for source in candidate_sources
                if all(
                    (
                        target_node := roadmap.transition_node(source, orbit)
                    ) is not None
                    and any(
                        parent is desired_next_orbit
                        for parent in target_node.parent_orbit
                    )
                    for roadmap in constituent_roadmaps.values()
                )
            ]

        if (
            not guided_sources
            and desired_modal_state is not None
            and modal_graph is not None
        ):
            guided_sources = [
                source
                for source in candidate_sources
                if modal_graph.transition_advances_toward(
                    orbit,
                    source,
                    desired_modal_state,
                )
            ]

        # Use one probability draw so a bias of 0.8 still leaves exactly 0.2
        # probability for unrestricted transition exploration.
        if guided_sources and random.random() < modal_transition_bias:
            candidate_sources = guided_sources

        return random.choice(candidate_sources)


    @staticmethod
    def sample_roadmap_configuration(
        orbit: Orbit,
        constituent_roadmaps: dict[int, ConstituentRoadmap],
    ) -> Configuration | None:
        """Sample an existing point from the implicit composite roadmap."""
        joint_angles = []
        for roadmap in constituent_roadmaps.values():
            candidates = [
                node
                for node in roadmap.nodes_in_orbit(orbit)
                if not node.is_transition
            ]
            if not candidates:
                return None
            joint_angles.append(random.choice(candidates).joint_angles)
        return Configuration(
            joint_angles=joint_angles,
            attachments=dict(orbit.attachments),
            is_transition=False,
            parent_orbit=orbit,
        )


    @staticmethod
    def sample_goal_configuration(
        orbit: Orbit,
        constituent_roadmaps: dict[int, ConstituentRoadmap],
    ) -> Configuration | None:
        """Build a composite target from constituent goals in ``orbit``."""
        joint_angles = []
        for roadmap in constituent_roadmaps.values():
            candidates = [
                node
                for node in roadmap.goal_nodes
                if any(parent is orbit for parent in node.parent_orbit)
            ]
            if not candidates:
                return None
            joint_angles.append(random.choice(candidates).joint_angles)

        return Configuration(
            joint_angles=joint_angles,
            attachments=dict(orbit.attachments),
            is_transition=False,
            parent_orbit=orbit,
        )


    def retrace_path(self, node: TensorProductNode) -> list[TensorProductNode]:
        """
        Retrace the path from the given node back to the root of the tree.

        :param node: The TensorProductNode to start retracing from.
        :return: A list of TensorProductNodes representing the path from root to the given node.
        """
        path = []
        current_node = node
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.parent
        return path[::-1]  # Return reversed path


    def to_composite_path(
        self,
        path: list[TensorProductNode] | None,
    ) -> tuple[list[Configuration] | None, float]:
        """Convert a tensor path and insert pre-poses around transitions."""
        if path is None:
            return None, float("inf")

        configurations = [
            node.to_composite_configuration()
            for node in path
        ]
        expanded_path = []

        for index, configuration in enumerate(configurations):
            if not configuration.is_transition or not configuration.pre_pose:
                expanded_path.append(configuration)
                continue

            previous_configuration = (
                configurations[index - 1]
                if index > 0
                else None
            )
            next_configuration = (
                configurations[index + 1]
                if index + 1 < len(configurations)
                else None
            )

            incoming_orbit = TensorProductTree._shared_orbit(
                configuration,
                previous_configuration,
            )
            if incoming_orbit is not None and previous_configuration is not None:
                expanded_path.append(Configuration(
                    joint_angles=copy.deepcopy(configuration.pre_pose),
                    is_transition=False,
                    parent_orbit=incoming_orbit,
                ))

            expanded_path.append(configuration)

            outgoing_orbit = TensorProductTree._shared_orbit(
                configuration,
                next_configuration,
            )
            if outgoing_orbit is not None and next_configuration is not None:
                expanded_path.append(Configuration(
                    joint_angles=copy.deepcopy(configuration.pre_pose),
                    is_transition=False,
                    parent_orbit=outgoing_orbit,
                ))

        cost = sum(
            TensorProductTree.compute_distance_between_configurations(a, b)
            for a, b in zip(expanded_path, expanded_path[1:])
        )
        return expanded_path, cost


    @staticmethod
    def _shared_orbit(
        transition: Configuration,
        adjacent: Configuration | None,
    ) -> Orbit | None:
        if adjacent is None or not transition.parent_orbit or not adjacent.parent_orbit:
            return None
        return next(
            (
                orbit
                for orbit in transition.parent_orbit
                if orbit in adjacent.parent_orbit
            ),
            None,
        )


    @staticmethod
    def compute_distance_between_configurations(config_a, config_b):
        if len(config_a.joint_angles) != len(config_b.joint_angles):
            raise ValueError("Configurations must contain the same number of robots.")
        return sum(
            np.linalg.norm(np.asarray(joints_a) - np.asarray(joints_b))
            for joints_a, joints_b in zip(
                config_a.joint_angles,
                config_b.joint_angles,
            )
        )


    def find_nearest_node(self, configuration: Configuration, orbit: Orbit) -> TensorProductNode:
        """
        Find the nearest node in the tree to the given configuration within the specified orbit.

        :param configuration: The composite configuration to find the nearest node to.
        :param orbit: The orbit to search within.
        :return: The nearest TensorProductNode in the tree.
        """
        nearest_node = None
        min_distance = float('inf')
        
        for node_key in self._ordered_orbit_node_keys(orbit):
            node = self.nodes[node_key]
            distance = self.compute_distance(node.joint_angles, configuration.joint_angles)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        return nearest_node


    def find_nearest_reachable_transition_node(
        self,
        transition: Configuration,
        orbit: Orbit,
        constituent_roadmaps: dict[int, ConstituentRoadmap],
        nearest_node: TensorProductNode | None = None,
    ) -> TensorProductNode | None:
        """Return the nearest tensor node that can reach ``transition``.

        The ordinary Euclidean-nearest node is accepted immediately when all
        of its constituent nodes occur in the target distance maps. Only when
        that node is disconnected do we scan the remaining tensor nodes in
        the orbit for one that is. When no tensor node in the orbit can reach
        every constituent target yet, we fall back to the plain
        Euclidean-nearest node instead of abandoning the extension step:
        ``informed_extend`` can still make per-robot progress toward the
        transition through whichever roadmap edges already exist, which is
        how reachability develops in the first place. Returning ``None``
        here discards the whole expansion step, and doing that whenever no
        node is fully reachable yet starves the search of the very
        transitions it needs to explore new orbits.
        """
        target_distances = []
        for roadmap in constituent_roadmaps.values():
            target_node = roadmap.transition_node(transition, orbit)
            if target_node is None:
                return nearest_node
            distances = roadmap.distances_to_target(target_node, orbit)
            if not distances:
                return nearest_node
            target_distances.append(distances)

        def is_reachable(node):
            return len(node.constituent_nodes) == len(target_distances) and all(
                component in distances
                for component, distances in zip(
                    node.constituent_nodes,
                    target_distances,
                )
            )

        if nearest_node is not None and is_reachable(nearest_node):
            return nearest_node

        reachable_nearest = None
        nearest_distance = float("inf")
        for node_key in self._ordered_orbit_node_keys(orbit):
            candidate = self.nodes[node_key]
            if candidate is nearest_node or not is_reachable(candidate):
                continue
            distance = self.compute_distance(
                candidate.joint_angles,
                transition.joint_angles,
            )
            if distance < nearest_distance:
                nearest_distance = distance
                reachable_nearest = candidate
        return reachable_nearest if reachable_nearest is not None else nearest_node


    def get_neighbors(
        self,
        new_node: TensorProductNode,
        orbit: Orbit,
        constituent_roadmaps: dict[int, ConstituentRoadmap],
    ) -> list[TensorProductNode]:
        """
        Find all the neighbors of the new node within the specified orbit.

        :param new_node: The new TensorProductNode to find neighbors for.
        :param orbit: The orbit to search within.
        :return: A list of neighbor TensorProductNodes.
        """
        neighbors = []
        for node_key in self._ordered_orbit_node_keys(orbit):
            if node_key != new_node.identity_key:
                candidate = self.nodes[node_key]
                if self.is_implicit_tensor_edge(
                    candidate,
                    new_node,
                    orbit,
                    constituent_roadmaps,
                ):
                    neighbors.append(candidate)
        return heapq.nsmallest(
            min(orbit.k, len(neighbors)),
            neighbors,
            key=lambda candidate: (
                self.compute_distance(
                    candidate.joint_angles,
                    new_node.joint_angles,
                ),
                candidate.creation_index,
            ),
        )


    @staticmethod
    def is_implicit_tensor_edge(
        node_a: TensorProductNode,
        node_b: TensorProductNode,
        orbit: Orbit,
        constituent_roadmaps: dict[int, ConstituentRoadmap],
    ) -> bool:
        """Check that every robot waits or follows an available roadmap edge."""
        roadmaps = list(constituent_roadmaps.values())
        if not (
            len(node_a.constituent_nodes)
            == len(node_b.constituent_nodes)
            == len(roadmaps)
        ):
            return False

        moved = False
        for roadmap, component_a, component_b in zip(
            roadmaps,
            node_a.constituent_nodes,
            node_b.constituent_nodes,
        ):
            if component_a is component_b:
                continue
            moved = True
            if not roadmap.has_edge_in_orbit(
                component_a,
                component_b,
                orbit,
            ):
                return False
        return moved


    def choose_best_parent(self, new_node: TensorProductNode, neighbors: list[TensorProductNode], parent_orbit: Orbit) -> TensorProductNode:
        """
        Choose the best parent for the new node from its neighbors based on cost and collision checking.

        :param new_node: The new TensorProductNode to connect.
        :param neighbors: A list of neighbor TensorProductNodes.
        :param parent_orbit: The orbit of the parent node for collision checking.
        :return: The best parent TensorProductNode or None if no valid parent is found.
        """
        candidates = sorted(
            neighbors,
            key=lambda neighbor: (
                neighbor.cost
                + self.estimate_edge_cost(neighbor, new_node),
                neighbor.creation_index,
            ),
        )

        # UNKNOWN and VALID edges use the same optimistic geometric cost;
        # INVALID edges return infinity. The first usable candidate is thus
        # the cheapest optimistic parent. Level-2 validation is deferred until
        # a resulting root-to-goal path can improve the incumbent.
        for neighbor in candidates:
            edge_cost = self.compute_cost(neighbor, new_node, parent_orbit)
            if np.isfinite(edge_cost):
                return neighbor
        return None


    def add_node(self, new_node, parent_node=None, parent_orbit=None):
        """
        Add a new node to the tree and register it in the orbit_nodes dictionary.

        :param new_node: The new TensorProductNode to add.
        :param parent_node: The parent TensorProductNode to connect to (optional).
        """

        new_node.creation_index = self._next_node_index
        self._next_node_index += 1
        self.nodes[new_node.identity_key] = new_node
        for orbit in new_node.parent_orbit:
            orbit_key = id(orbit)
            if orbit_key not in self.orbit_nodes:
                self.orbit_nodes[orbit_key] = set()
                self._orbit_node_order[orbit_key] = []
            self.orbit_nodes[orbit_key].add(new_node.identity_key)
            self._orbit_node_order[orbit_key].append(new_node.identity_key)

        new_node.parent = parent_node
        new_node.parent_edge_orbit = parent_orbit if parent_node is not None else None
        if parent_node is not None:
            parent_node.children.append(new_node)
            new_node.cost = parent_node.cost + self.compute_cost(parent_node, new_node, parent_orbit)


    def _ordered_orbit_node_keys(self, orbit: Orbit):
        """Return tensor-node keys in per-tree deterministic creation order."""
        return self._orbit_node_order.get(id(orbit), ())


    def rewire_node(self, node, new_parent, orbit: Orbit):
        """
        Rewire the new node to a new parent if it results in a lower cost.

        :param node: The TensorProductNode to rewire.
        :param new_parent: The new parent TensorProductNode to connect to.
        """
        if node is self.root or node is new_parent:
            return False

        ancestor = new_parent
        while ancestor is not None:
            if ancestor is node:
                return False
            ancestor = ancestor.parent

        estimated_cost = new_parent.cost + self.estimate_edge_cost(
            new_parent,
            node,
        )
        if estimated_cost >= node.cost:
            return False

        edge_cost = self.compute_cost(new_parent, node, orbit)
        new_cost = new_parent.cost + edge_cost
        if not np.isfinite(new_cost) or new_cost >= node.cost:
            return False

        old_cost = node.cost
        if node.parent is not None:
            node.parent.children.remove(node)
        node.parent = new_parent
        node.parent_edge_orbit = orbit
        if node not in new_parent.children:
            new_parent.children.append(node)
        node.cost = new_cost

        cost_delta = new_cost - old_cost
        descendants = list(node.children)
        while descendants:
            descendant = descendants.pop()
            descendant.cost += cost_delta
            descendants.extend(descendant.children)

        self.successful_rewires += 1
        return True


    def estimate_edge_cost(
        self,
        parent_node: TensorProductNode,
        child_node: TensorProductNode,
    ) -> float:
        """Cheap exact cost for a collision-free tensor-product edge."""
        edge_cost = self.compute_distance(
            self.motion_joint_angles(parent_node),
            self.motion_joint_angles(child_node),
        )
        if self.is_transition_configuration(child_node):
            edge_cost += self.transition_action_cost(child_node)
        return edge_cost


    def compute_cost(self, parent_node: TensorProductNode, child_node: TensorProductNode, orbit: Orbit):
        """
        Return an optimistic tensor-edge cost.

        Level-1 constituent connectivity remains mandatory. Level-2
        inter-robot collision checking is deferred until this edge appears on
        a candidate goal path. A previously invalidated edge has infinite
        cost and can never be selected again.

        :param parent_node: The parent TensorProductNode.
        :param child_node: The child TensorProductNode.
        :return: The cost of moving from the parent to the child.
        """
        if (
            self.constituent_roadmaps is not None
            and not self.is_implicit_tensor_edge(
                parent_node,
                child_node,
                orbit,
                self.constituent_roadmaps,
            )
        ):
            return float('inf')

        if self.tensor_edge_status(parent_node, child_node, orbit) == self.EDGE_INVALID:
            return float('inf')
        return self.estimate_edge_cost(parent_node, child_node)


    @staticmethod
    def _tensor_edge_key(
        node_a: TensorProductNode,
        node_b: TensorProductNode,
        orbit: Orbit,
    ):
        """Identity key for symmetric Level-2 collision validity."""
        return (
            id(orbit),
            frozenset((node_a.identity_key, node_b.identity_key)),
        )


    def tensor_edge_status(
        self,
        node_a: TensorProductNode,
        node_b: TensorProductNode,
        orbit: Orbit,
    ):
        return self._tensor_edge_status.get(
            self._tensor_edge_key(node_a, node_b, orbit),
            self.EDGE_UNKNOWN,
        )


    def validate_tensor_edge(
        self,
        node_a: TensorProductNode,
        node_b: TensorProductNode,
        orbit: Orbit,
    ) -> bool:
        """Validate one UNKNOWN Level-2 edge and cache the result."""
        # Even a cached Level-2 result does not establish Level-1 validity.
        if not self._validate_constituent_motion(node_a, node_b, orbit):
            return False
        edge_key = self._tensor_edge_key(node_a, node_b, orbit)
        status = self._tensor_edge_status.get(edge_key, self.EDGE_UNKNOWN)
        if status == self.EDGE_VALID:
            return True
        if status == self.EDGE_INVALID:
            return False

        if (
            self.constituent_roadmaps is not None
            and not self.is_implicit_tensor_edge(
                node_a,
                node_b,
                orbit,
                self.constituent_roadmaps,
            )
        ):
            # Constituent roadmap topology may gain edges later, so do not
            # permanently blacklist a pair merely because Level 1 currently
            # has no connecting edge.
            return False

        start_configuration = Configuration(
            joint_angles=self.motion_joint_angles(node_a),
            is_transition=False,
            parent_orbit=orbit,
        )
        goal_configuration = Configuration(
            joint_angles=self.motion_joint_angles(node_b),
            is_transition=False,
            parent_orbit=orbit,
        )
        self.lazy_edge_validation_checks += 1
        is_valid = self.environment.tensor_product_motion_collision_checking(
            start_configuration,
            goal_configuration,
            orbit,
        )
        self._tensor_edge_status[edge_key] = (
            self.EDGE_VALID if is_valid else self.EDGE_INVALID
        )
        if not is_valid:
            self.lazy_invalid_edges += 1
        return is_valid


    @staticmethod
    def motion_joint_angles(node: TensorProductNode):
        """Use transition pre-poses as the endpoints of roadmap motion."""
        return [
            constituent.pre_pose
            if constituent.is_transition and constituent.pre_pose is not None
            else constituent.joint_angles
            for constituent in node.constituent_nodes
        ]


    @staticmethod
    def transition_action_cost(node: TensorProductNode) -> float:
        """Cost of approaching and retreating from a complete transition."""
        return sum(
            2.0 * np.linalg.norm(
                np.asarray(constituent.pre_pose)
                - np.asarray(constituent.joint_angles)
            )
            for constituent in node.constituent_nodes
            if constituent.pre_pose is not None
        )


    def compute_distance(self, config1: list, config2: list):
        """
        Compute the distance between two configurations = sum of Euclidean distances between corresponding joint angles.

        :param config1: The first list of joint angles.
        :param config2: The second list of joint angles.
        :return: The distance between the two list of configurations.
        """
        if len(config1) != len(config2):
            raise ValueError("Configuration dimensions do not match.")
        else:
            return sum(np.linalg.norm(np.array(joint_angle1) - np.array(joint_angle2)) for joint_angle1, joint_angle2 in zip(config1, config2))


    def compute_composite_heuristic(self, node: TensorProductNode):
        """
        Compute the composite heuristic value for the given node based on its constituent roadmap nodes.

        :param node: The TensorProductNode to compute the heuristic for.
        :return: The composite heuristic value.
        """
        values = [
            roadmap_node.heuristic_value
            for roadmap_node in node.constituent_nodes
        ]
        if any(value is None for value in values):
            return float('inf')
        return sum(values)


    def is_transition_configuration(self, node: TensorProductNode):
        """
        Check if the given node is a transition configuration.

        :param node: The TensorProductNode to check.
        :return: True if the node is a transition configuration, False otherwise.
        """
        transition_keys = {
            id(roadmap_node._transition_key)
            for roadmap_node in node.constituent_nodes
            if roadmap_node.is_transition
            and roadmap_node._transition_key is not None
        }
        return (
            bool(node.constituent_nodes)
            and all(
                roadmap_node.is_transition
                for roadmap_node in node.constituent_nodes
            )
            and len(transition_keys) == 1
        )
    

    def get_next_orbit(self, node: TensorProductNode, current_orbit: Orbit):
        """
        Get candidate next orbits shared by all constituent roadmap nodes.

        :param node: TensorProductNode to check.
        :param current_orbit: Current orbit.
        :return: List of candidate next orbits.
        """

        if not node.constituent_nodes:
            return []
        # Preserve the first constituent node's deterministic parent-orbit
        # order instead of returning list(set(...)).
        return [
            orbit
            for orbit in node.constituent_nodes[0].parent_orbit
            if orbit is not current_orbit
            and all(
                orbit in roadmap_node.parent_orbit
                for roadmap_node in node.constituent_nodes[1:]
            )
        ]


    def goal_reached(self, node: TensorProductNode):
        """
        Check whether the current composite node itself is a goal.

        :param node: The TensorProductNode to check.
        :return: True if the node has reached the goal, False otherwise.
        """
        return (
            bool(node.constituent_nodes)
            and all(
                roadmap_node.is_goal
                for roadmap_node in node.constituent_nodes
            )
            and any(
                orbit.is_goal_orbit
                for orbit in node.parent_orbit
            )
        )


    def try_connect_goal(
        self,
        node: TensorProductNode,
        orbit: Orbit,
        constituent_roadmaps: dict[int, ConstituentRoadmap],
    ) -> TensorProductNode | None:
        """Connect ``node`` to an exact tensor goal through one implicit edge."""
        if not orbit.is_goal_orbit or orbit not in node.parent_orbit:
            return None

        if self.goal_reached(node):
            self.reached_goal_nodes.add(node)
            return node

        roadmaps = list(constituent_roadmaps.values())
        if len(node.constituent_nodes) != len(roadmaps):
            return None

        goal_components = []
        for roadmap, current_component in zip(
            roadmaps,
            node.constituent_nodes,
        ):
            reachable_goals = [
                goal
                for goal in roadmap.goal_nodes
                if orbit in goal.parent_orbit
                and (
                    goal is current_component
                    or roadmap.has_edge_in_orbit(
                        current_component,
                        goal,
                        orbit,
                    )
                )
            ]
            if not reachable_goals:
                return None

            goal_components.append(min(
                reachable_goals,
                key=lambda goal: (
                    np.linalg.norm(
                        np.asarray(current_component.joint_angles)
                        - np.asarray(goal.joint_angles)
                    ),
                    goal.creation_index,
                ),
            ))

        tensor_goal = TensorProductNode(
            constituent_nodes=goal_components,
            parent_orbit=orbit,
        )
        existing_goal = self.nodes.get(tensor_goal.identity_key)

        if existing_goal is node:
            self.reached_goal_nodes.add(node)
            return node

        if (
            existing_goal is not None
            and node.cost + self.estimate_edge_cost(node, existing_goal)
            >= existing_goal.cost
        ):
            self.reached_goal_nodes.add(existing_goal)
            return existing_goal

        edge_cost = self.compute_cost(node, tensor_goal, orbit)
        if not np.isfinite(edge_cost):
            return None

        if existing_goal is None:
            self.add_node(
                tensor_goal,
                parent_node=node,
                parent_orbit=orbit,
            )
            existing_goal = tensor_goal
        elif node.cost + edge_cost < existing_goal.cost:
            self.rewire_node(existing_goal, node, orbit)

        self.reached_goal_nodes.add(existing_goal)
        return existing_goal


    def _get_information(self):
        """
        Get information about the given node for debugging or visualization purposes.

        :param node: The TensorProductNode to get information about.
        :return: A dictionary containing information about the node.
        """
        info = {
            "lazy_level1_checks": sum(
                getattr(roadmap, "lazy_collision_checks", 0)
                for roadmap in (self.constituent_roadmaps or {}).values()
            ),
            "invalid_level1_contexts": sum(
                getattr(roadmap, "lazy_invalid_edges", 0)
                for roadmap in (self.constituent_roadmaps or {}).values()
            ),
            "target_distance_searches": sum(
                getattr(roadmap, "target_distance_searches", 0)
                for roadmap in (self.constituent_roadmaps or {}).values()
            ),
            "target_distance_cache_hits": sum(
                getattr(roadmap, "target_distance_cache_hits", 0)
                for roadmap in (self.constituent_roadmaps or {}).values()
            ),
            "num_nodes": len(self.nodes),
            "successful_rewires": self.successful_rewires,
            "num_explored_orbits": len(self.orbit_experience),
            "lazy_edge_validation_checks": self.lazy_edge_validation_checks,
            "lazy_invalid_edges": self.lazy_invalid_edges,
            "known_valid_tensor_edges": sum(
                status == self.EDGE_VALID
                for status in self._tensor_edge_status.values()
            ),
            "known_invalid_tensor_edges": sum(
                status == self.EDGE_INVALID
                for status in self._tensor_edge_status.values()
            ),
        }
        return info
    

def roulette_select(candidates, score):
    """Sample candidates with lower minimization scores receiving more weight."""
    if not candidates:
        return None

    scores = np.asarray([score(candidate) for candidate in candidates], dtype=float)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return random.choice(candidates)

    minimum = np.min(scores[finite])
    spread = np.max(scores[finite]) - minimum
    if spread <= 1e-12:
        weights = finite.astype(float)
    else:
        weights = np.zeros_like(scores)
        weights[finite] = np.exp(-(scores[finite] - minimum) / spread)

    return random.choices(candidates, weights=weights, k=1)[0]


def greedy_or_roulette_select(candidates, score, greedy_probability=0.8):
    """Usually take the best candidate while retaining stochastic exploration."""
    if random.random() < greedy_probability:
        return min(candidates, key=score)
    return roulette_select(candidates, score)


def informed_extend(
    V_near: TensorProductNode,
    Q_rand: Configuration,
    O_selection: Orbit,
    constituent_roadmaps: dict[int, ConstituentRoadmap],
    environment=None,
) -> TensorProductNode | None:
    
    """
    Extend the tree towards a random configuration while considering heuristic values.

    :param V_near: The nearest node in the tree to the random configuration.
    :param Q_rand: The random configuration to extend towards.
    :param O_selection: The selected orbit for the extension.
    :param constituent_roadmaps: A dictionary of constituent roadmaps for each robot.
    :param environment: Reserved for compatibility; roadmap edges were already
        available in the current orbit; optional Level-1 checks are deferred.
    :return: The new node added to the tree, or None if the extension failed.
    """

    if len(V_near.constituent_nodes) != len(constituent_roadmaps):
        raise ValueError(
            "V_near must contain one constituent node per roadmap."
        )

    candidate_sets = []
    for component_index, (_, con_roadmap) in enumerate(
        constituent_roadmaps.items()
    ):
        current_node = V_near.constituent_nodes[component_index]
        candidates = con_roadmap.neighbors(current_node, O_selection)
        candidates.append(current_node)     # Ai = NeighborsInOrbit(v_i_near) ∪ {v_i_near}
        candidate_sets.append(candidates)

    if Q_rand.is_transition:
        new_constituent_nodes = []
        for (_, roadmap), candidates in zip(
            constituent_roadmaps.items(),
            candidate_sets,
        ):
            target_node = roadmap.transition_node(Q_rand, O_selection)
            if target_node is None:
                return None
            distances = roadmap.distances_to_target(target_node, O_selection)
            reachable_candidates = [
                candidate
                for candidate in candidates
                if candidate in distances
                and (
                    not candidate.is_transition
                    or candidate._transition_key is Q_rand
                )
            ]
            if not reachable_candidates:
                return None
            new_constituent_nodes.append(greedy_or_roulette_select(
                reachable_candidates,
                distances.__getitem__,
            ))
    else:
        new_constituent_nodes = []
        for component_index, candidates in enumerate(candidate_sets):
            ordinary_candidates = [
                candidate
                for candidate in candidates
                if not candidate.is_transition
            ]
            if not ordinary_candidates:
                return None
            selected_node = greedy_or_roulette_select(
                ordinary_candidates,
                lambda node: np.linalg.norm(
                    np.asarray(node.joint_angles)
                    - np.asarray(Q_rand.joint_angles[component_index])
                ),
            )
            new_constituent_nodes.append(selected_node)

    if all(
        selected is current
        for selected, current in zip(
            new_constituent_nodes,
            V_near.constituent_nodes,
        )
    ):
        return None

    transition_keys = {
        id(node._transition_key)
        for node in new_constituent_nodes
        if node.is_transition and node._transition_key is not None
    }
    if len(transition_keys) > 1:
        return None

    common_orbits = list(new_constituent_nodes[0].parent_orbit)
    for node in new_constituent_nodes[1:]:
        common_orbits = [
            orbit
            for orbit in common_orbits
            if orbit in node.parent_orbit
        ]
    if O_selection not in common_orbits:
        return None

    return TensorProductNode(
        constituent_nodes=new_constituent_nodes,
        parent_orbit=common_orbits,
    )
