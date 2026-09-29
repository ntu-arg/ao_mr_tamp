
from collections import defaultdict, deque
import random
import time
import numpy as np
import copy
import heapq

from planners.prm_roadmap_utils import (
    Configuration,
    ModalAction,
    ModalState,
    Mode,
    Orbit,
)


class ModalGraph:
    """Canonical high-level projection of the discovered orbit graph.

    Modes are canonicalized by attachment signature. Modal states are
    canonicalized by ``(mode, object-at-goal status)``. Several geometrically
    distinct orbits may therefore be associated with the same modal state.
    """

    def __init__(self, environment):
        self.environment = environment
        self.robot_ids = tuple(
            sorted(
                robot.id
                for robot in getattr(environment, "robots", [])
            )
        )
        self.object_ids = tuple(
            sorted(
                obj.id
                for obj in getattr(environment, "mobjects", [])
            )
        )
        self.object_index = {
            object_id: index
            for index, object_id in enumerate(self.object_ids)
        }

        self.modes_by_key = {}
        self.states_by_key = {}
        self.modal_graph = {}
        self.orbits_by_state = defaultdict(set)
        self.state_by_orbit = {}
        self.realized_actions = defaultdict(set)
        self.expanded_states = set()
        self.initial_state = None
        self.goal_states = set()
        self.preferred_path = []

    def get_or_add_mode(self, attachments):
        """Return the canonical mode for an attachment mapping/signature."""
        candidate = self._mode_from_attachments(attachments)
        canonical = self.modes_by_key.get(candidate)
        if canonical is None:
            self.modes_by_key[candidate] = candidate
            canonical = candidate
        return canonical

    def _mode_from_attachments(self, attachments):
        return Mode.from_attachments(
            attachments,
            robot_ids=self.robot_ids or None,
        )

    def has_mode(self, attachments):
        """Return whether an attachment mode has already been discovered."""
        return self._mode_from_attachments(attachments) in self.modes_by_key

    def _goal_state_key(self):
        _, _, goal_attachments = self.environment._get_goal_conditions()
        goal_mode = self._mode_from_attachments(goal_attachments)
        goal_progress = tuple(True for _ in self.object_ids)
        return ModalState(mode=goal_mode, at_goal=goal_progress)

    def has_goal_state(self):
        """Return whether the complete target modal state was discovered."""
        return self._goal_state_key() in self.states_by_key

    def get_or_add_state(self, mode, at_goal):
        """Return the canonical modal state for mode and task progress."""
        canonical_mode = self.modes_by_key.get(mode)
        if canonical_mode is None:
            self.modes_by_key[mode] = mode
            canonical_mode = mode
        candidate = ModalState(mode=canonical_mode, at_goal=tuple(at_goal))
        canonical = self.states_by_key.get(candidate)
        if canonical is None:
            self.states_by_key[candidate] = candidate
            self.modal_graph[candidate] = set()
            canonical = candidate
        return canonical

    def _object_goal_status(self, orbit):
        status_function = getattr(
            self.environment,
            "get_object_goal_status",
            None,
        )
        if status_function is None:
            # Environments without placement-style task progress still get a
            # valid modal representation. They can override this hook before
            # modal guidance is enabled.
            object_ids = sorted(
                obj.id
                for obj in getattr(self.environment, "mobjects", [])
            )
            return tuple(False for _ in object_ids)

        status = tuple(bool(value) for value in status_function(orbit))
        expected_size = len(getattr(self.environment, "mobjects", []))
        if len(status) != expected_size:
            raise ValueError(
                "Object goal-status vector must contain one value per object."
            )
        return status

    def modal_state_for_orbit(self, orbit):
        """Build a modal-state value without registering the orbit or state."""
        return ModalState(
            mode=self._mode_from_attachments(orbit.attachments),
            at_goal=self._object_goal_status(orbit),
        )

    def modal_state_after_transition(self, orbit, transition_configuration):
        """Return the modal state reached by a validated boundary sample."""
        cache_key = (id(self), id(orbit))
        projection_cache = getattr(
            transition_configuration,
            "_modal_projection_cache",
            None,
        )
        if projection_cache is not None:
            cached = projection_cache.get(cache_key)
            if cached is not None and cached[0] is self and cached[1] is orbit:
                return cached[2]
        neighboring_orbit = orbit.define_neighboring_orbit(
            transition_configuration
        )
        state = self.modal_state_for_orbit(neighboring_orbit)
        if projection_cache is None:
            projection_cache = {}
            transition_configuration._modal_projection_cache = projection_cache
        projection_cache[cache_key] = (self, orbit, state)
        return state

    @staticmethod
    def advances_toward_modal_state(current_state, candidate_state, target_state):
        """Return whether a candidate contains the requested modal progress.

        Components changed by ``target_state`` must be reached exactly. Other
        robots may transition concurrently, but already completed object goals
        may not be undone. This lets a multi-robot transition satisfy a
        one-action modal target without treating the modal prediction as a
        geometric validity test.
        """
        changed = False
        for current, target, candidate in zip(
            current_state.mode.attachments,
            target_state.mode.attachments,
            candidate_state.mode.attachments,
        ):
            if target != current:
                changed = True
                if candidate != target:
                    return False

        for current, target, candidate in zip(
            current_state.at_goal,
            target_state.at_goal,
            candidate_state.at_goal,
        ):
            if current and not candidate:
                return False
            if target != current:
                changed = True
                if candidate != target:
                    return False
        return changed

    @staticmethod
    def action_toward_modal_state(current_state, target_state):
        """Return one pick/place action required by a modal target.

        The returned action is only a sampling hint. The environment still
        performs IK and collision validation, and a separate unrestricted
        sampling branch preserves probabilistic completeness.
        """
        if current_state is None or target_state is None:
            return None

        robot_ids = tuple(range(len(current_state.mode.attachments)))
        for robot_id, current, target in zip(
            robot_ids,
            current_state.mode.attachments,
            target_state.mode.attachments,
        ):
            if current == target:
                continue
            if current is None and target is not None:
                return ModalAction("pick", robot_id, target)
            if current is not None and target is None:
                return ModalAction("place", robot_id, current)

        # Placement can change task progress without changing attachments.
        for object_id, current, target in zip(
            range(len(current_state.at_goal)),
            current_state.at_goal,
            target_state.at_goal,
        ):
            if not current and target:
                for robot_id, attachment in zip(
                    robot_ids,
                    current_state.mode.attachments,
                ):
                    if attachment == object_id:
                        return ModalAction("place", robot_id, object_id)
        return None

    def transition_advances_toward(
        self,
        orbit,
        transition_configuration,
        target_state,
    ):
        candidate_state = self.modal_state_after_transition(
            orbit,
            transition_configuration,
        )
        return self.advances_toward_modal_state(
            orbit.parent_modal_state,
            candidate_state,
            target_state,
        )

    def register_orbit(self, orbit, *, is_initial=False, is_goal=None):
        """Associate a canonical geometric orbit with canonical modal data."""
        existing_state = self.state_by_orbit.get(orbit)
        if existing_state is not None:
            if is_initial:
                self.initial_state = existing_state
            if is_goal:
                self.goal_states.add(existing_state)
            return existing_state

        mode = self.get_or_add_mode(orbit.attachments)
        state = self.get_or_add_state(mode, self._object_goal_status(orbit))

        orbit.parent_mode = mode
        orbit.parent_modal_state = state
        self.state_by_orbit[orbit] = state
        self.orbits_by_state[state].add(orbit)

        if is_initial:
            self.initial_state = state
        if is_goal is None:
            is_goal = orbit.is_goal(environment=self.environment)
        if is_goal:
            self.goal_states.add(state)
        return state

    def _goal_attachments_are_free(self):
        return all(
            attachment is None
            for attachment in self._goal_state_key().mode.attachments
        )

    def _modal_pick_candidates(self, state, robot_id, available_objects):
        candidate_function = getattr(
            self.environment,
            "get_modal_pick_candidates",
            None,
        )
        if candidate_function is None:
            return list(available_objects)
        return list(
            candidate_function(
                state,
                robot_id,
                list(available_objects),
            )
        )

    def _modal_place_is_allowed(self, state, robot_id, object_id):
        predicate = getattr(
            self.environment,
            "is_modal_place_allowed",
            None,
        )
        if predicate is None:
            return True
        return bool(predicate(state, robot_id, object_id))

    def symbolic_successors(self, state):
        """Generate inexpensive pick/place successors without claiming feasibility."""
        if not self._goal_attachments_are_free():
            # Attachment-goal tasks require environment-specific symbolic
            # semantics and remain entirely unguided for now.
            return []

        attachments = list(state.mode.attachments)
        held_objects = {
            object_id
            for object_id in attachments
            if object_id is not None
        }
        incomplete_objects = [
            object_id
            for object_id, complete in zip(self.object_ids, state.at_goal)
            if not complete and object_id not in held_objects
        ]
        successors = []

        for robot_index, (robot_id, held_object) in enumerate(
            zip(self.robot_ids, attachments)
        ):
            if held_object is None:
                pick_candidates = self._modal_pick_candidates(
                    state,
                    robot_id,
                    incomplete_objects,
                )
                for object_id in pick_candidates:
                    if object_id not in incomplete_objects:
                        continue
                    next_attachments = list(attachments)
                    next_attachments[robot_index] = object_id
                    next_state = ModalState(
                        mode=Mode(tuple(next_attachments)),
                        at_goal=state.at_goal,
                    )
                    successors.append(
                        (next_state, (ModalAction("pick", robot_id, object_id),))
                    )
                continue

            if not self._modal_place_is_allowed(
                state,
                robot_id,
                held_object,
            ):
                continue
            object_index = self.object_index.get(held_object)
            if object_index is None:
                continue
            next_attachments = list(attachments)
            next_attachments[robot_index] = None
            next_progress = list(state.at_goal)
            next_progress[object_index] = True
            next_state = ModalState(
                mode=Mode(tuple(next_attachments)),
                at_goal=tuple(next_progress),
            )
            successors.append(
                (next_state, (ModalAction("place", robot_id, held_object),))
            )

        return successors

    def symbolic_goal_distance(self, state):
        """Optimistic remaining pick/place count for free-attachment goals."""
        if not self._goal_attachments_are_free():
            return float("inf")
        held_objects = {
            object_id
            for object_id in state.mode.attachments
            if object_id is not None
        }
        distance = 0
        for object_id, complete in zip(self.object_ids, state.at_goal):
            if complete:
                continue
            distance += 1 if object_id in held_objects else 2
        return distance

    def modal_state_cost(self, state):
        """Return the environment-aware cost used for modal guidance.

        Environments may encode task-specific constraints, such as stacking
        precedence, without putting those rules in the general planner.  The
        generic optimistic pick/place distance remains the fallback.
        """
        cost_function = getattr(
            self.environment,
            "get_modal_state_cost",
            None,
        )
        if cost_function is None:
            return self.symbolic_goal_distance(state)
        return float(cost_function(state))

    def optimistic_modal_target(self, state):
        """Select one optimistic, goal-progressing symbolic successor.

        The returned state is guidance only. It is not registered as a
        realized modal state until a valid boundary transition creates a
        corresponding orbit.
        """
        current_distance = self.modal_state_cost(state)
        candidates = [
            next_state
            for next_state, _ in self.symbolic_successors(state)
            if self.modal_state_cost(next_state) < current_distance
        ]
        if not candidates:
            return None

        best_distance = min(
            self.modal_state_cost(candidate)
            for candidate in candidates
        )
        best_candidates = [
            candidate
            for candidate in candidates
            if self.modal_state_cost(candidate) == best_distance
        ]
        return random.choice(best_candidates)

    def goal_progressing_targets(self, state):
        """Return all one-step symbolic successors that reduce goal distance."""
        current_distance = self.modal_state_cost(state)
        return [
            next_state
            for next_state, _ in self.symbolic_successors(state)
            if self.modal_state_cost(next_state) < current_distance
        ]

    def preferred_path_from(self, state):
        """Construct one strictly goal-progressing symbolic modal path."""
        goal_state = self._goal_state_key()
        if state == goal_state:
            return [state]

        path = [state]
        current = state
        maximum_steps = max(1, 2 * len(self.object_ids) + len(self.robot_ids))
        for _ in range(maximum_steps):
            current_distance = self.modal_state_cost(current)
            candidates = [
                (next_state, actions)
                for next_state, actions in self.symbolic_successors(current)
                if self.modal_state_cost(next_state) < current_distance
            ]
            if not candidates:
                return []

            best_distance = min(
                self.modal_state_cost(next_state)
                for next_state, _ in candidates
            )
            best_candidates = [
                candidate
                for candidate in candidates
                if self.modal_state_cost(candidate[0]) == best_distance
            ]
            current, _ = random.choice(best_candidates)
            path.append(current)
            if current == goal_state:
                self.preferred_path = path
                return path
        return []

    def select_guided_orbit(
        self,
        orbits,
        temperature=1.0,
        state_exploration_probability=0.1,
    ):
        """Select a modal state first, then an orbit within that state.

        This prevents a modal state from receiving more probability merely
        because it has accumulated more geometric orbit realizations.
        """
        if not 0.0 <= state_exploration_probability <= 1.0:
            raise ValueError(
                "state_exploration_probability must be in [0, 1]."
            )

        candidate_orbits_by_state = defaultdict(list)
        for orbit in orbits:
            state = orbit.parent_modal_state
            if state is None:
                continue
            candidate_orbits_by_state[state].append(orbit)

        if not candidate_orbits_by_state:
            return None, None, []

        # Evaluate each canonical modal state exactly once, regardless of how
        # many geometric orbits currently realize it.
        state_costs = {}
        for state in list(candidate_orbits_by_state):
            cost = self.modal_state_cost(state)
            if (
                not np.isfinite(cost)
                or cost <= 0
                or not self.symbolic_successors(state)
            ):
                del candidate_orbits_by_state[state]
                continue
            state_costs[state] = cost

        if not candidate_orbits_by_state:
            return None, None, []

        states = list(candidate_orbits_by_state)
        costs = [state_costs[state] for state in states]
        minimum_cost = min(costs)
        scale = max(float(temperature), 1e-9)
        exploitation_weights = np.asarray(
            [np.exp(-(cost - minimum_cost) / scale) for cost in costs],
            dtype=float,
        )
        exploitation_weights /= np.sum(exploitation_weights)
        state_count = len(states)
        weights = [
            (
                (1.0 - state_exploration_probability) * weight
                + state_exploration_probability / state_count
            )
            for weight in exploitation_weights
        ]
        state = random.choices(states, weights=weights, k=1)[0]
        orbit = random.choice(candidate_orbits_by_state[state])
        path = self.preferred_path_from(state)
        if len(path) < 2:
            return None, None, []
        self.preferred_path = path
        return orbit, path[1], path

    def _infer_actions(self, from_mode, to_mode):
        """Infer directed pick/place labels from attachment changes."""
        actions = []
        robot_ids = self.robot_ids or tuple(range(len(from_mode.attachments)))
        for robot_id, before, after in zip(
            robot_ids,
            from_mode.attachments,
            to_mode.attachments,
        ):
            if before == after:
                continue
            if before is not None:
                actions.append(ModalAction("place", robot_id, before))
            if after is not None:
                actions.append(ModalAction("pick", robot_id, after))
        return tuple(actions)

    def _add_directed_realized_transition(self, from_state, to_state):
        if from_state == to_state:
            return
        self.modal_graph[from_state].add(to_state)
        actions = self._infer_actions(from_state.mode, to_state.mode)
        self.realized_actions[(from_state, to_state)].add(actions)

    def add_realized_transition(self, from_orbit, to_orbit):
        """Project an undirected orbital edge into two directed modal edges."""
        from_state = self.state_by_orbit[from_orbit]
        to_state = self.state_by_orbit[to_orbit]
        self._add_directed_realized_transition(from_state, to_state)
        self._add_directed_realized_transition(to_state, from_state)

    def get_neighbors(self, state):
        return self.modal_graph.get(state, set())

    def is_goal_state_reachable(self):
        """Return whether a goal modal state is reachable from the start."""
        if self.initial_state is None:
            return False

        goal_state = self.states_by_key.get(self._goal_state_key())
        if goal_state is None:
            return False

        visited = {self.initial_state}
        queue = deque([self.initial_state])
        while queue:
            state = queue.popleft()
            if state == goal_state:
                return True
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def mark_expanded(self, state):
        """Track symbolic expansion per modal state, never per pure mode."""
        canonical = self.states_by_key.get(state)
        if canonical is None:
            raise KeyError("Modal state is not registered in this graph.")
        self.expanded_states.add(canonical)

    def __len__(self):
        return len(self.modal_graph)



class OrbitGraph:
    def __init__(self, q_s, init_orbit_states, environment):
        self.environment = environment
        self.orbit_graph = {}
        self._next_orbit_index = 0
        self.modal_graph = ModalGraph(environment)
        # initialize the graph with the initial configuration
        initial_orbit = Orbit(
            configuration=q_s,
            init_states=init_orbit_states,
            environment=environment
        )
        self.initial_orbit = initial_orbit
        self.goal_orbits = []
        self.add_node(initial_orbit)
        q_s.add_parent_orbit(initial_orbit)


    def add_node(self, node):
        if node in self.orbit_graph:
            for existing_node in self.orbit_graph:
                if existing_node == node:
                    node.parent_mode = existing_node.parent_mode
                    node.parent_modal_state = existing_node.parent_modal_state
                    return existing_node
        is_initial = len(self.orbit_graph) == 0
        node.creation_index = self._next_orbit_index
        self._next_orbit_index += 1
        self.orbit_graph[node] = []
        # check if an orbit is goal whenever we add them
        is_goal = node.is_goal(environment=node.environment)
        if is_goal and node not in self.goal_orbits:
            self.goal_orbits.append(node)
        self.modal_graph.register_orbit(
            node,
            is_initial=is_initial,
            is_goal=is_goal,
        )
        return node


    # Edges have equal weight
    def add_edge(self, from_node, to_node):
        if from_node in self.orbit_graph and to_node in self.orbit_graph:
            if to_node not in self.orbit_graph[from_node]:
                self.orbit_graph[from_node].append(to_node)
            if from_node not in self.orbit_graph[to_node]:
                self.orbit_graph[to_node].append(from_node)
            self.modal_graph.add_realized_transition(from_node, to_node)


    def get_neighbors(self, node):
        return self.orbit_graph.get(node, [])

    def has_found_goal_orbit(self):
        """Return whether a complete goal orbit has been discovered."""
        return bool(self.goal_orbits)

    def has_found_goal_attachment_mode(self):
        """Return whether the goal attachment signature has been discovered."""
        _, _, goal_attachments = self.environment._get_goal_conditions()
        return self.modal_graph.has_mode(goal_attachments)

    def has_found_goal_mode(self):
        """Backward-compatible alias for the attachment-only mode check."""
        return self.has_found_goal_attachment_mode()

    def has_found_goal_modal_state(self):
        """Return whether the complete goal modal state has been discovered."""
        return self.modal_graph.has_goal_state()


    def uniform_sample_orbit(self) -> Orbit:
        ''' 
            Sample an orbit from the graph (uniform sampling).
        '''
        return random.choice(list(self.orbit_graph.keys()))


    def biased_sample_orbit(self) -> Orbit:
        """Compatibility alias; modal guidance now supplies the search bias."""
        return self.uniform_sample_orbit()


    def is_goal_orbit_reachable(self):
        """Return whether any goal orbit is reachable from O_S."""
        if not self.orbit_graph or self.initial_orbit not in self.orbit_graph:
            return False

        visited = {self.initial_orbit}
        queue = deque([self.initial_orbit])

        while queue:
            current_orbit = queue.popleft()

            if current_orbit.is_goal(environment=current_orbit.environment):
                return True

            for neighbor in self.get_neighbors(current_orbit):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False  


    def get_goal_orbits(self):
        """Return all currently discovered orbits that satisfy the goal."""
        self.goal_orbits = [
            orbit
            for orbit in self.orbit_graph
            if orbit.is_goal(environment=orbit.environment)
        ]
        return list(self.goal_orbits)


    def best_goal_orbit_path(self):
        """Return the shortest realized orbit path from the start to a goal."""
        goal_orbits = set(self.get_goal_orbits())
        if not goal_orbits or self.initial_orbit not in self.orbit_graph:
            return []

        parents = {self.initial_orbit: None}
        queue = deque([self.initial_orbit])
        reached_goal = None
        while queue:
            orbit = queue.popleft()
            if orbit in goal_orbits:
                reached_goal = orbit
                break
            for neighbor in self.orbit_graph[orbit]:
                if neighbor in parents:
                    continue
                parents[neighbor] = orbit
                queue.append(neighbor)

        if reached_goal is None:
            return []

        path = []
        orbit = reached_goal
        while orbit is not None:
            path.append(orbit)
            orbit = parents[orbit]
        return list(reversed(path))


    def __str__(self):
        return str(self.orbit_graph)


    def update_heuristic_values(self):
        """
        Update all orbital heuristics with one multi-source breadth-first search.
        """
        self.goal_orbits = [
            orbit
            for orbit in self.orbit_graph
            if orbit.is_goal(environment=orbit.environment)
        ]
        for orbit in self.orbit_graph:
            orbit.heuristic_value = float('inf')

        queue = deque(self.goal_orbits)
        for goal_orbit in self.goal_orbits:
            goal_orbit.heuristic_value = 0

        while queue:
            orbit = queue.popleft()
            candidate = orbit.heuristic_value + 1
            for neighbor in self.orbit_graph[orbit]:
                if candidate >= neighbor.heuristic_value:
                    continue
                neighbor.heuristic_value = candidate
                queue.append(neighbor)


    def orbit_dijkstra(self, start_orbit: Orbit, target_orbit: Orbit) -> dict:
        """
            Finds the shortest path in a RoadmapGraph.
            'roadmap_graph' is a dict of dicts:
            {Configuration: {Configuration: distance, ...}, ...}

            Returns (distances, parent_map). Use reconstruct_path() to get the path.
                distances: dict mapping each node to its distance from start_node
        """
        distances = {node: float('inf') for node in self.orbit_graph}
        distances[start_orbit] = 0

        # Use a counter as tiebreaker so heapq never compares Configuration objects
        counter = 0
        priority_queue = [(0, counter, start_orbit)]     # cost, counter, node

        while priority_queue:
            current_cost, _, current_node = heapq.heappop(priority_queue)   # choose the node with the lowest cost

            if current_node is target_orbit:
                return distances

            if current_cost > distances[current_node]:
                continue

            for neighbor in self.orbit_graph[current_node]:
                new_cost = current_cost + 1  # all edges have equal weight

                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost      # update cost
                    counter += 1
                    heapq.heappush(priority_queue, (new_cost, counter, neighbor))

        return distances
    



class RoadmapGraph:
    def __init__(self):
        self.graph = {}
        # Equal transition samples must resolve to the same object so their
        # parent-orbit memberships can be accumulated over time.
        self._canonical_nodes = {}
        self._connected_orbits_by_node = {}
        self._edge_collision_cache = {}
        # A geometric edge is only usable in the orbit context where it was
        # constructed and collision checked.
        self._edge_orbits = {}

        self.best_goal_cost = float('inf')  # Track the best goal cost found so far
        self.best_goal_node = None          # Track the best goal node found so far


    def _collision_free_cached(self, from_node, to_node, orbit, environment):
        if environment is None:
            return True
        key = (id(orbit) if orbit is not None else -1, frozenset((from_node, to_node)))
        if key in self._edge_collision_cache:
            return self._edge_collision_cache[key]
        is_ok = collision_free(from_node, to_node, orbit, environment)
        self._edge_collision_cache[key] = is_ok
        return is_ok
    
        
    def add_interior_node(self, node, orbit, environment=None):
        canonical, was_added = self._get_or_add_node(node)
        if was_added:
            self._connect_node_in_orbit(canonical, orbit, environment)
        return canonical


    def _get_or_add_node(self, node):
        canonical = self._canonical_nodes.get(node)
        if canonical is not None:
            return canonical, False

        self._canonical_nodes[node] = node
        self.graph[node] = {}
        self._connected_orbits_by_node[node] = set()
        return node, True


    def _ensure_node_in_orbit(self, node, orbit):
        """Make the orbit's configuration list reference the canonical node."""
        for index, configuration in enumerate(orbit.configurations):
            if configuration == node:
                if configuration is not node:
                    orbit.configurations[index] = node
                    orbit.update_kd_tree()
                return

        orbit.configurations.append(node)
        orbit.update_kd_tree()


    def _connect_node_in_orbit(self, node, orbit, environment):
        orbit_key = id(orbit)
        connected_orbits = self._connected_orbits_by_node.setdefault(node, set())
        if orbit_key in connected_orbits:
            return

        self._ensure_node_in_orbit(node, orbit)
        for neighbor in orbit.nearest_neighbors(node):
            canonical_neighbor, _ = self._get_or_add_node(neighbor)
            if canonical_neighbor == node:
                continue
            if not self._collision_free_cached(
                node, canonical_neighbor, orbit, environment
            ):
                continue

            edge_distance = distance(node, canonical_neighbor)
            self._record_edge_in_orbit(
                node, canonical_neighbor, edge_distance, orbit
            )

        connected_orbits.add(orbit_key)


    def _record_edge_in_orbit(self, node_a, node_b, edge_distance, orbit):
        """Record an undirected roadmap edge together with its orbit label."""
        self.graph.setdefault(node_a, {})
        self.graph.setdefault(node_b, {})
        self.graph[node_a][node_b] = min(
            edge_distance,
            self.graph[node_a].get(node_b, float("inf")),
        )
        self.graph[node_b][node_a] = self.graph[node_a][node_b]
        edge_key = frozenset((node_a, node_b))
        self._edge_orbits.setdefault(edge_key, {})[id(orbit)] = orbit


    def remove_node(self, node):
        if node not in self.graph:
            return
        # Remove this node from all neighbors
        for neighbor in list(self.graph[node].keys()):
            if neighbor in self.graph:
                self.graph[neighbor].pop(node, None)
        # Remove the node itself
        del self.graph[node]
        for edge_key in [
            edge_key
            for edge_key in self._edge_orbits
            if node in edge_key
        ]:
            self._edge_orbits.pop(edge_key, None)
        self._canonical_nodes.pop(node, None)
        self._connected_orbits_by_node.pop(node, None)


    def expand_connectivity(self, node, orbit, environment=None):
        if node not in self.graph:
            return
        
        neighbors = orbit.nearest_neighbors(node)
        for neig in neighbors:
            if neig == node:
                continue
            canonical_neighbor, _ = self._get_or_add_node(neig)
            if not self._collision_free_cached(
                node, canonical_neighbor, orbit, environment
            ):
                continue
            self._record_edge_in_orbit(
                node,
                canonical_neighbor,
                distance(node, canonical_neighbor),
                orbit,
            )


    def add_boundary_node(self, node, parent_orbits, environment=None):
        """Add a transition in its explicit incident-orbit context.

        The same transition data may occur at a boundary between different
        orbit pairs. Keep those as separate graph states so global Dijkstra
        cannot splice together edges from unrelated task contexts. A boundary
        incident to more than two orbits is represented by one contextual
        bridge state per pair passed by the caller; discovering additional
        successor orbits remains the caller's responsibility.
        """
        if not node.is_transition:
            raise ValueError("Boundary nodes must be transition configurations.")

        unique_orbits = tuple(
            {id(orbit): orbit for orbit in parent_orbits}.values()
        )
        if not unique_orbits:
            raise ValueError("A transition roadmap node needs an incident orbit.")

        # A node's graph identity includes this orbit context. Clone instead
        # of mutating ``node`` because it may already be referenced by an
        # orbit's sampled-configuration list, or may be reused for another
        # valid boundary context by a caller.
        contextual_node = copy.copy(node)
        # The caller supplies the authoritative order: source orbit first,
        # destination orbit second. Do not inherit stale membership/order from
        # a sampled Configuration that may have been reused.
        contextual_node.parent_orbit = list(unique_orbits)
        contextual_node.roadmap_context = tuple(
            id(orbit) for orbit in unique_orbits
        )

        # Replace the sampled transition reference in each incident orbit so
        # nearest-neighbor queries see the same context-aware graph identity.
        for orbit in unique_orbits:
            replaced = False
            for index, configuration in enumerate(orbit.configurations):
                if configuration is node:
                    orbit.configurations[index] = contextual_node
                    replaced = True
            if replaced:
                orbit.update_kd_tree()

        canonical, _ = self._get_or_add_node(contextual_node)

        for orbit in unique_orbits:
            canonical.add_parent_orbit(orbit)
            self._connect_node_in_orbit(canonical, orbit, environment)

        return canonical
    

    def add_edge(self, from_node, to_node, orbit=None, environment=None):
        if from_node in self.graph and to_node in self.graph:
            if not self._collision_free_cached(from_node, to_node, orbit, environment):
                return
            dis = distance(from_node, to_node)
            if orbit is None:
                if to_node in self.graph[from_node]:
                    return
                self.graph[from_node][to_node] = dis
                self.graph[to_node][from_node] = dis
            else:
                self._record_edge_in_orbit(from_node, to_node, dis, orbit)


    def get_neighbors(self, node):
        return list(self.graph.get(node, {}).keys())


    def _dijkstra_in_orbit_context(self, start_node, start_orbit, goal_states):
        """Search roadmap states as ``(configuration, orbit)`` pairs.

        Motion edges remain inside their labeled orbit.  The only way to
        change orbit context is across a transition node's explicit ordered
        source/destination pair.
        """
        adjacency = {}

        def add_arc(source, target, cost):
            adjacency.setdefault(source, {})
            adjacency.setdefault(target, {})
            old_cost = adjacency[source].get(target, float("inf"))
            if cost < old_cost:
                adjacency[source][target] = cost

        # Intra-orbit motion edges.  Do not infer context from mere node
        # membership; require that this edge was checked in that orbit.
        for node_a, neighbors in self.graph.items():
            for node_b in neighbors:
                edge_key = frozenset((node_a, node_b))
                edge_contexts = self._edge_orbits.get(edge_key, {})
                for orbit in edge_contexts.values():
                    state_a = (node_a, id(orbit))
                    state_b = (node_b, id(orbit))
                    add_arc(state_a, state_b, self.graph[node_a][node_b])

        # Directed task-state changes.  A transition generated from A creates
        # an explicit A -> B arc; its shared geometry alone cannot switch
        # contexts or return to A as though the action had not happened.
        for transition in self.graph:
            if not transition.is_transition:
                continue
            incident_orbits = list(transition.parent_orbit or ())
            if len(incident_orbits) < 2:
                continue
            source_orbit, destination_orbit = incident_orbits[:2]
            add_arc(
                (transition, id(source_orbit)),
                (transition, id(destination_orbit)),
                0.0,
            )

        start_state = (start_node, id(start_orbit))
        distances = {start_state: 0.0}
        parents = {}
        queue = [(0.0, 0, start_state)]
        insertion_order = 1
        while queue:
            current_distance, _, current_state = heapq.heappop(queue)
            if current_distance != distances.get(current_state):
                continue
            for neighbor_state, edge_cost in adjacency.get(
                current_state, {}
            ).items():
                candidate_distance = current_distance + edge_cost
                if candidate_distance < distances.get(
                    neighbor_state, float("inf")
                ):
                    distances[neighbor_state] = candidate_distance
                    parents[neighbor_state] = current_state
                    heapq.heappush(
                        queue,
                        (candidate_distance, insertion_order, neighbor_state),
                    )
                    insertion_order += 1

        reachable_goals = [
            (distances[(goal, id(orbit))], (goal, id(orbit)))
            for goal, orbit in goal_states
            if (goal, id(orbit)) in distances
        ]
        if not reachable_goals:
            return distances, parents, None, None

        best_cost, best_state = min(reachable_goals, key=lambda item: item[0])
        state_path = [best_state]
        while state_path[-1] != start_state:
            parent = parents.get(state_path[-1])
            if parent is None:
                return distances, parents, None, None
            state_path.append(parent)
        state_path.reverse()

        # Collapse the two context states of a crossed transition into one
        # action node. If a transition is only used as a same-orbit motion
        # waypoint, emit a normal configuration in that orbit instead.
        path = []
        index = 0
        while index < len(state_path):
            node, orbit_id = state_path[index]
            run_end = index
            while (
                run_end + 1 < len(state_path)
                and state_path[run_end + 1][0] is node
            ):
                run_end += 1

            if node.is_transition:
                before_orbit_id = (
                    state_path[index - 1][1] if index > 0 else None
                )
                after_orbit_id = (
                    state_path[run_end + 1][1]
                    if run_end + 1 < len(state_path)
                    else None
                )
                if (
                    before_orbit_id is not None
                    and after_orbit_id is not None
                    and before_orbit_id != after_orbit_id
                ):
                    path.append(node)
                else:
                    orbit = next(
                        (
                            candidate
                            for candidate in node.parent_orbit or ()
                            if id(candidate) == orbit_id
                        ),
                        None,
                    )
                    motion_pose = node.pre_pose or node.joint_angles
                    path.append(
                        Configuration(
                            joint_angles=copy.deepcopy(motion_pose),
                            attachments=(
                                dict(orbit.attachments) if orbit is not None else {}
                            ),
                            is_transition=False,
                            parent_orbit=orbit,
                        )
                    )
            else:
                path.append(node)
            index = run_end + 1

        return distances, parents, (best_cost, best_state), path
    
    
    def retrace_path(self, orbit_graph, start_node, env):

        """
            :return: duration (for computation time checking), path (list of configurations), best_cost
        """

        start_time = time.perf_counter()

        # Ensure the start configuration is represented in the roadmap.
        start_orbits = getattr(start_node, "parent_orbit", None) or []
        if not start_orbits:
            print("Start configuration has no associated orbit; cannot search orbit-context roadmap.")
            duration = time.perf_counter() - start_time
            return duration, None, None
        if start_node not in self.graph:
            # normally start_node is an interior node so it has one parent orbit
            self.add_interior_node(start_node, start_orbits[0], environment=env)
        for start_orb in start_orbits:
            self.expand_connectivity(start_node, start_orb, environment=env)

        # find orbit of goal
        goal_orbits = []
        for orb in orbit_graph.orbit_graph:
            if orb.is_goal(environment=env):
                goal_orbits.append(orb)
        if len(goal_orbits) == 0:
            print("No goal orbit found in orbital graph.")
            duration = time.perf_counter() - start_time
            return duration, None, None
        

        # Add goal configuration candidates to roadmap if not already there, connect to neighbors.
        # When robot poses are unconstrained, evaluate all valid interior nodes in goal orbits
        # instead of a single random sample so Dijkstra cannot miss reachable goal components.
        robot_poses, object_poses, attachments = env._get_goal_conditions()
        goal_states = []
        for goal_orb in goal_orbits:
            if len(goal_orb.goal_configurations) > 0:
                goal_states.extend(
                    (goal_config, goal_orb)
                    for goal_config in goal_orb.goal_configurations
                )
                for goal_config in goal_orb.goal_configurations:
                    if goal_config not in self.graph:
                        self.add_interior_node(goal_config, goal_orb, environment=env)
                    else:
                        self.expand_connectivity(goal_config, goal_orb, environment=env)
            elif robot_poses:
                goal_configuration = Configuration(
                    joint_angles=robot_poses,
                    attachments=attachments,
                    is_transition=False,
                    parent_orbit=goal_orb
                )
                goal_configuration = goal_orb.add_goal_configuration(goal_configuration)
                goal_states.append((goal_configuration, goal_orb))
                if goal_configuration not in self.graph:
                    self.add_interior_node(goal_configuration, goal_orb, environment=env)
                else:
                    # Keep expanding goal-node connectivity as the roadmap grows.
                    # Otherwise, the goal can get "stuck" with early sparse neighbors.
                    self.expand_connectivity(goal_configuration, goal_orb, environment=env)
            else:
                for cfg in goal_orb.configurations:
                    if cfg.is_transition:
                        continue
                    if cfg.attachments != attachments:
                        continue
                    goal_states.append((cfg, goal_orb))
                    if cfg not in self.graph:
                        self.add_interior_node(cfg, goal_orb, environment=env)
                    else:
                        self.expand_connectivity(cfg, goal_orb, environment=env)
                    if len(self.graph.get(cfg, {})) == 0:
                        self.expand_connectivity(cfg, goal_orb, environment=env)

        # Search only paths whose motion edges remain in one orbit and whose
        # task-state changes cross an explicit transition bridge.
        distances, parent_map, best_goal_result, path = (
            self._dijkstra_in_orbit_context(
                start_node,
                start_orbits[0],
                goal_states,
            )
        )
        if best_goal_result is None or path is None:
            print("No path to any goal configuration found.")
            duration = time.perf_counter() - start_time
            return duration, None, None
        best_cost, best_goal_state = best_goal_result
        self.best_goal_cost = best_cost
        self.best_goal_node = best_goal_state[0]

        for count, node in enumerate(path):
            print(f"-> Node {count}")
            print(f"Node attachments: {node.attachments}. Is transition: {node.is_transition}")
            if node.grasp_poses:
                for obj_id, grasp_pose in node.grasp_poses.items():
                    print(f"Grasp pose for object {obj_id}: {grasp_pose}")
            if node.placement_poses:
                for obj_id, placement_pose in node.placement_poses.items():
                    print(f"Placement pose for object {obj_id}: {placement_pose}")
            if node.pre_pose:
                for r_id, pre_pose in enumerate(node.pre_pose):
                    print(f"Pre-pose for robot {r_id}: {pre_pose}")


        
        # Post-process the path:
        # For every transition configuration, there is prepose to reach before and after performing grasp/ placement
        # Create a interior configuration object for the prepose and insert it into the path
        current_orbit = path[0].parent_orbit[0] if path[0].parent_orbit else None
        new_path = []
        for index, configuration in enumerate(path):

            if configuration.is_transition and configuration.pre_pose and index > 0:
                # Insert pre_pose configuration before the transition
                # TODO: for the robot which does not perform grasp/placement, the pre_pose now is same with transition random pose
                # but it will makes the robot to stop and waiting while other robot is going from pre_pose to transition pose
                # it's better to do interpolation from the previous pose on these robot to let it not waiting.

                previous_joint_angles = path[index - 1].joint_angles if not path[index-1].is_transition else path[index-1].pre_pose
                pre_pose_value = copy.deepcopy(configuration.pre_pose)
                for r_id, robot_pre_pose in enumerate(pre_pose_value):
                    robot_transition_pose = configuration.joint_angles[r_id]
                    if np.allclose(robot_pre_pose, robot_transition_pose, atol=0.01):
                        # The pre_pose is the interpolation of transition pose and the previous pose
                        avg_joint_angles = (np.array(robot_pre_pose) + np.array(previous_joint_angles[r_id])) / 2
                        pre_pose_value[r_id] = avg_joint_angles.tolist()

                pre_pose_configuration = Configuration(
                    joint_angles=pre_pose_value,
                    is_transition=False,
                    parent_orbit=current_orbit
                )
                new_path.append(pre_pose_configuration)

            new_path.append(configuration)

            # next orbit and after pose configuration
            if configuration.is_transition and configuration.pre_pose and index + 1 < len(path):
                next_orbit = next((orbit for orbit in configuration.parent_orbit if orbit in path[index+1].parent_orbit),
                                   None)
                if next_orbit:
                    current_orbit = next_orbit
                    # Insert pre_pose configuration after the transition
                    # interpolate the pre_pose and the next configuration to avoid waiting

                    after_pose_value = copy.deepcopy(configuration.pre_pose)
                    next_joint_angles = path[index + 1].joint_angles if not path[index+1].is_transition else path[index+1].pre_pose
                    for r_id, robot_after_pose in enumerate(after_pose_value):
                        robot_transition_pose = configuration.joint_angles[r_id]
                        if np.allclose(robot_after_pose, robot_transition_pose, atol=0.01):
                            # The after_pose is the interpolation of transition pose and the next pose
                            avg_joint_angles = (np.array(robot_after_pose) + np.array(next_joint_angles[r_id])) / 2
                            after_pose_value[r_id] = avg_joint_angles.tolist()

                    pre_pose_configuration_after = Configuration(
                        joint_angles=after_pose_value,
                        is_transition=False,
                        parent_orbit=current_orbit
                    )
                    new_path.append(pre_pose_configuration_after)

        
        # Report the cost of the executable, post-processed path. This keeps
        # PRM and dRRT* on the same metric and includes each transition's
        # approach/retreat exactly once.
        executable_cost = path_distance(new_path)
        duration = time.perf_counter() - start_time
        return duration, new_path, executable_cost
    

    def __str__(self):
        return str(self.graph)



def distance(config1, config2):
    """
    Computes the separate joint-space travel lengths for an arbitrary number of robots and sums them up.

    if config1 and config2 are both interior nodes:
        distance = |config1.joint_angles - config2.joint_angles|
    Transition action cost is split equally across the two roadmap edges
    incident to a transition. Therefore, a complete path segment
    interior -> pre-pose -> action -> pre-pose -> interior counts the
    approach/retreat motion exactly once.

    if config1 and config2 are both transition nodes:
        distance = |config1.pre_pose - config1.joint_angles| + |config1.pre_pose - config2.pre_pose| + |config2.joint_angles - config2.pre_pose|
    if config1 is an interior node and config2 is a transition node:
        distance = |config1.joint_angles - config2.pre_pose| + |config2.joint_angles - config2.pre_pose|
    if config1 is a transition node and config2 is an interior node:
        distance = |config1.pre_pose - config1.joint_angles| + |config1.pre_pose - config2.joint_angles|
    
    Expects joint_angles to be a nested iterable: [[robot1_joints], [robot2_joints], ...]
    """
    if len(config1.joint_angles) != len(config2.joint_angles):
        raise ValueError("The number of robots in both configurations must match")

    total_travel_distance = 0.0

    if (not config1.is_transition and not config2.is_transition):
        for joints_r1, joints_r2 in zip(config1.joint_angles, config2.joint_angles):
            # Convert lists to numpy arrays to handle the vector subtraction cleanly
            qr1 = np.array(joints_r1)
            qr2 = np.array(joints_r2)
            if len(qr1) != len(qr2):
                raise ValueError("Joint angle profiles for the same robot must have identical lengths")
            total_travel_distance += np.linalg.norm(qr1 - qr2)
        return total_travel_distance
    
    elif (config1.is_transition and config2.is_transition):
        for joints_r1, joints_r2, pre_pose_r1, pre_pose_r2 in zip(config1.joint_angles, config2.joint_angles, config1.pre_pose, config2.pre_pose):
            qr1 = np.array(joints_r1)
            qr2 = np.array(joints_r2)
            pre_qr1 = np.array(pre_pose_r1)
            pre_qr2 = np.array(pre_pose_r2)

            total_travel_distance += np.linalg.norm(pre_qr1 - qr1) + np.linalg.norm(pre_qr1 - pre_qr2) + np.linalg.norm(qr2 - pre_qr2)
        return total_travel_distance
    
    elif (not config1.is_transition and config2.is_transition):
        for joints_r1, joints_r2, pre_pose_r2 in zip(config1.joint_angles, config2.joint_angles, config2.pre_pose):
            qr1 = np.array(joints_r1)
            qr2 = np.array(joints_r2)
            pre_qr2 = np.array(pre_pose_r2)

            total_travel_distance += np.linalg.norm(qr1 - pre_qr2) + np.linalg.norm(qr2 - pre_qr2)
        return total_travel_distance
    
    elif (config1.is_transition and not config2.is_transition):
        for joints_r1, joints_r2, pre_pose_r1 in zip(config1.joint_angles, config2.joint_angles, config1.pre_pose):
            qr1 = np.array(joints_r1)
            qr2 = np.array(joints_r2)
            pre_qr1 = np.array(pre_pose_r1)

            total_travel_distance += np.linalg.norm(pre_qr1 - qr1) + np.linalg.norm(pre_qr1 - qr2)
        return total_travel_distance
    
    else:
        raise ValueError("Unsupported configuration types")


def path_distance(path):
    """Return joint-space length of an executable configuration path."""
    if not path or len(path) < 2:
        return 0.0

    total = 0.0
    for config1, config2 in zip(path, path[1:]):
        if len(config1.joint_angles) != len(config2.joint_angles):
            raise ValueError(
                "Adjacent configurations must contain the same number of robots."
            )
        total += sum(
            np.linalg.norm(
                np.asarray(joints1, dtype=float)
                - np.asarray(joints2, dtype=float)
            )
            for joints1, joints2 in zip(
                config1.joint_angles,
                config2.joint_angles,
            )
        )
    return total




def _pre_pose_configuration(transition_configuration, orbit):
    """Represent a transition pre-pose in the common configuration format.

    Most environments historically accepted raw joint-angle arrays in
    ``collision_free_motion``.  Some environments, such as handover, use
    the configuration's attachments while interpolating and therefore require
    a ``Configuration`` object.  Normalize the pre-pose at this planner
    boundary so all environments receive the same interface.
    """
    joint_angles = transition_configuration.pre_pose
    if joint_angles is None or len(joint_angles) == 0:
        joint_angles = transition_configuration.joint_angles
    return Configuration(
        joint_angles=joint_angles,
        attachments=transition_configuration.attachments,
        is_transition=False,
        parent_orbit=orbit,
    )


def collision_free(config1, config2, orbit, environment, num_checks=None) -> bool:
    '''
        Check if the straight-line path between two configurations
        is free of obstacle collisions by adaptive midpoint checks.
    '''

    # if one of them is a transition node, we need to check the pre_pose to joint_angles segment
    if (config1.is_transition and not config2.is_transition):
        pre_pose_configuration = _pre_pose_configuration(config1, orbit)

        return environment.collision_free_motion(
            pre_pose_configuration,
            config2,
            orbit=orbit,
            # resolution=resolution,
            num_checks=num_checks,
        )
    
    if (not config1.is_transition and config2.is_transition):
        pre_pose_configuration = _pre_pose_configuration(config2, orbit)

        return environment.collision_free_motion(
            config1,
            pre_pose_configuration,
            orbit=orbit,
            # resolution=resolution,
            num_checks=num_checks,
        )
    
    if (config1.is_transition and config2.is_transition):
        pre_pose_config1 = _pre_pose_configuration(config1, orbit)
        pre_pose_config2 = _pre_pose_configuration(config2, orbit)

        return environment.collision_free_motion(
            pre_pose_config1,
            pre_pose_config2,
            orbit=orbit,
            # resolution=resolution,
            num_checks=num_checks,
        )


    return environment.collision_free_motion(
        config1,
        config2,
        orbit=orbit,
        # resolution=resolution,
        num_checks=num_checks,
    )
