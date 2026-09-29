"""Lazy PRM* roadmap implementation for the task-and-motion planner.

The existing :class:`prm_roadmap_utils.Orbit` supplies the PRM* k-nearest
neighborhood (``k = k0 * log(n)``).  This module changes only when roadmap
edges are collision checked: candidate edges are inserted optimistically and
checked only when they occur on a shortest path.
"""

from enum import Enum, auto
import contextlib
import io
import time

from planners.prm_roadmap import OrbitGraph, RoadmapGraph, collision_free
from planners.prm_roadmap_utils import dijkstra, reconstruct_path


class EdgeStatus(Enum):
    """Collision-validation state of an undirected roadmap edge."""

    UNKNOWN = auto()
    VALID = auto()
    INVALID = auto()


class LazyRoadmapGraph(RoadmapGraph):
    """A drop-in lazy variant of the existing k-PRM* roadmap.

    Node insertion and neighbor selection are inherited from ``RoadmapGraph``.
    Calls that would normally validate an edge instead register it as an
    unchecked candidate.  During a query, invalid edges on the current
    shortest path are removed and the search is repeated until the returned
    path is fully validated or no path remains.
    """

    def __init__(self, print_path_details=False):
        super().__init__()
        self.print_path_details = print_path_details
        self._edge_status = {}
        self._edge_orbit = {}
        self.lazy_collision_checks = 0
        self.lazy_validation_cache_hits = 0
        self.invalid_edges_removed = 0
        self.invalid_orbit_contexts = 0
        self.lazy_replans = 0
        self.lazy_validation_time = 0.0
        self.lazy_search_time = 0.0

    @staticmethod
    def _edge_key(from_node, to_node):
        return frozenset((from_node, to_node))

    @classmethod
    def _edge_context_key(cls, from_node, to_node, orbit):
        """Identify an edge validation in one manipulation orbit."""
        return (
            id(orbit) if orbit is not None else -1,
            cls._edge_key(from_node, to_node),
        )

    @classmethod
    def _edge_key_from_context(cls, context_key):
        """Return an edge key for a context key or legacy edge-only key."""
        return context_key[1] if isinstance(context_key, tuple) else context_key

    def _contexts_for_edge(self, edge_key):
        return [
            context_key
            for context_key in self._edge_status
            if self._edge_key_from_context(context_key) == edge_key
        ]

    def _collision_free_cached(self, from_node, to_node, orbit, environment):
        """Register an edge without performing collision checking."""
        edge_key = self._edge_key(from_node, to_node)
        if self._edge_status.get(edge_key) is EdgeStatus.INVALID:
            return False
        context_key = self._edge_context_key(from_node, to_node, orbit)
        if self._edge_status.get(context_key) is EdgeStatus.INVALID:
            return False
        # Absence of an environment at insertion time must not turn an
        # unchecked edge into a permanently valid edge.
        self._edge_status.setdefault(context_key, EdgeStatus.UNKNOWN)
        self._edge_orbit.setdefault(context_key, orbit)
        return True

    def _invalidate_edge_context(self, from_node, to_node, orbit):
        """Invalidate one orbit context and remove an edge if none remain."""
        edge_key = self._edge_key(from_node, to_node)
        context_key = self._edge_context_key(from_node, to_node, orbit)
        self._edge_status[context_key] = EdgeStatus.INVALID
        self.invalid_orbit_contexts += 1

        contexts = self._contexts_for_edge(edge_key)
        if any(
            self._edge_status[key] is not EdgeStatus.INVALID
            for key in contexts
        ):
            return

        # Keep the legacy edge-only status as a conservative block for later
        # connectivity expansion after this edge has been rejected.
        self._edge_status[edge_key] = EdgeStatus.INVALID

        self.graph.get(from_node, {}).pop(to_node, None)
        self.graph.get(to_node, {}).pop(from_node, None)
        self.invalid_edges_removed += 1

    def remove_node(self, node):
        super().remove_node(node)
        # Invalid edges are absent from graph adjacency, so inspect all lazy
        # metadata rather than only the node's current neighbors.
        context_keys = [
            context_key
            for context_key in self._edge_status
            if node in self._edge_key_from_context(context_key)
        ]
        for context_key in context_keys:
            self._edge_status.pop(context_key, None)
            self._edge_orbit.pop(context_key, None)

    def _validate_edge(self, from_node, to_node, environment):
        if environment is None:
            raise ValueError("Lazy edge validation requires an environment.")

        edge_key = self._edge_key(from_node, to_node)
        contexts = self._contexts_for_edge(edge_key)
        if not contexts:
            raise KeyError("Roadmap edge has no registered lazy context.")

        if any(
            self._edge_status[key] is EdgeStatus.VALID
            for key in contexts
        ):
            self.lazy_validation_cache_hits += 1
            return True

        # An edge shared by several orbits is usable when it is valid in at
        # least one registered context. Never reuse one orbit's result in a
        # different attachment/placement context.
        for context_key in contexts:
            if self._edge_status[context_key] is EdgeStatus.INVALID:
                continue
            orbit = self._edge_orbit[context_key]
            self.lazy_collision_checks += 1
            if collision_free(from_node, to_node, orbit, environment):
                self._edge_status[context_key] = EdgeStatus.VALID
                return True
            self._invalidate_edge_context(from_node, to_node, orbit)

        return False

    def _validate_path(self, path, environment):
        """Validate a path, stopping as soon as it must be replanned."""
        start_time = time.perf_counter()
        try:
            for from_node, to_node in zip(path, path[1:]):
                if not self._validate_edge(from_node, to_node, environment):
                    return False
            return True
        finally:
            self.lazy_validation_time += time.perf_counter() - start_time

    def lazy_shortest_path(self, start_node, goal_nodes, environment):
        """Return the lowest-cost collision-free path to any supplied goal.

        This public query is useful independently of the orbital goal logic.
        It returns ``(path, cost)`` and performs collision checks lazily.
        """
        goals = set(goal_nodes)
        while goals:
            self.lazy_replans += 1
            search_start = time.perf_counter()
            distances, parents = dijkstra(self.graph, start_node)
            self.lazy_search_time += time.perf_counter() - search_start
            reachable = [goal for goal in goals
                         if distances.get(goal, float("inf")) < float("inf")]
            if not reachable:
                return None, None

            goal = min(reachable, key=distances.__getitem__)
            path = reconstruct_path(parents, start_node, goal)
            if self._validate_path(path, environment):
                return path, distances[goal]

    def retrace_path(self, orbit_graph, start_node, env):
        """Run the existing orbital query with lazy edge validation."""
        start_time = time.perf_counter()

        while True:
            self.lazy_replans += 1
            # The parent implementation caches a best goal.  An invalidated
            # edge may make that goal more expensive, so each lazy replan must
            # compare all currently reachable goals afresh.
            self.best_goal_cost = float("inf")
            self.best_goal_node = None
            if self.print_path_details:
                _, path, cost = super().retrace_path(
                    orbit_graph, start_node, env
                )
            else:
                # The eager implementation prints every node before the lazy
                # collision validation step. Suppress those candidate-path
                # dumps; prm_main_v2 still prints its regular cost summaries.
                with contextlib.redirect_stdout(io.StringIO()):
                    _, path, cost = super().retrace_path(
                        orbit_graph, start_node, env
                    )
            if not path:
                return time.perf_counter() - start_time, None, None

            # The parent inserts pre/after-pose configurations for execution.
            # Keep only actual roadmap vertices to recover the searched path.
            roadmap_path = [node for node in path if node in self.graph]
            if self._validate_path(roadmap_path, env):
                return time.perf_counter() - start_time, path, cost

    def lazy_statistics(self):
        """Return counters suitable for experiment reporting."""
        return {
            "lazy_collision_checks": self.lazy_collision_checks,
            "lazy_validation_cache_hits": self.lazy_validation_cache_hits,
            "lazy_invalid_orbit_contexts": self.invalid_orbit_contexts,
            "lazy_invalid_edges_removed": self.invalid_edges_removed,
            "lazy_replans": self.lazy_replans,
            "lazy_validation_time": self.lazy_validation_time,
            "lazy_search_time": self.lazy_search_time,
        }


__all__ = ["EdgeStatus", "LazyRoadmapGraph", "OrbitGraph"]
