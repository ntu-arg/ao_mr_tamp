# Run this in terminal before executing the code to avoid potential issues with deep recursion in path retracing:
# ulimit -s unlimited


import numpy as np
import copy
import time
import random
import sys

import pybullet as p


from env.simple_pick_place import SimplePickPlaceEnv
from env.small_box_pick_place import SmallBoxPickPlaceEnv
from env.shelf_pick_place import ShelfPickPlaceEnv
from env.handover_object import HandoverEnv
from env.stacking_object import StackingObjectsEnv
from env.simple_motion_planning import SimpleMotionPlanningWithSubgoalEnv
from env.four_robots_pick_place import MultipleRobotsPickPlaceEnv as FourRobotPickPlaceEnv
from env.three_robots_pick_place import ThreeRobotsPickPlaceEnv
from env.stacking_castle_objects import StackingCastleEnv


from planners.prm_roadmap import OrbitGraph
from planners.prm_roadmap_utils import Configuration as CompositeConfiguration
from planners.drrt_star_roadmap import ConstituentRoadmap
from planners.drrt_star_tensor_tree import TensorProductTree, TensorProductNode
from planners.utils import plot_path_cost, save_path_to_file, \
    save_orbit_graph_to_file, get_orbit_path_from_path, \
    save_trial_statistics, cleanup_between_environments as _cleanup_between_environments, \
    set_global_seed as _set_global_seed

import faulthandler
faulthandler.enable()


alg_name = "drrt_star"


def _configure_deterministic_pybullet(environment):
    """Stabilize collision-pair ordering for PyBullet-backed environments."""
    client_id = getattr(environment, "using_client", None)
    if client_id is None:
        return
    try:
        if p.isConnected(client_id):
            p.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=client_id,
            )
    except p.error:
        # Non-PyBullet benchmark environments do not need this setting.
        pass


def select_sampling_guidance(
    orbital_graph,
    orbit_path,
    *,
    orbit_path_probability=0.6,
    modal_path_probability=0.3,
):
    """Select outer-loop orbit and transition guidance.

    Ordered realized-orbit edges receive the strongest bias when an orbit
    path is available.  Otherwise the existing optimistic modal path is used.
    The remaining probability samples the orbital graph uniformly, preserving
    exploration.

    Returns ``(O_samp, O_next, S_next, preferred_modal_path)``.
    """
    if (
        orbit_path_probability < 0.0
        or modal_path_probability < 0.0
        or orbit_path_probability + modal_path_probability > 1.0
    ):
        raise ValueError("Invalid outer-guidance probabilities.")

    candidates = list(orbital_graph.orbit_graph)
    if not candidates:
        return None, None, None, []

    candidate_by_id = {id(orbit): orbit for orbit in candidates}
    path_edges = [
        index
        for index in range(max(0, len(orbit_path or ()) - 1))
        if id(orbit_path[index]) in candidate_by_id
        and id(orbit_path[index + 1]) in candidate_by_id
        and any(
            neighbor is orbit_path[index + 1]
            for neighbor in orbital_graph.get_neighbors(orbit_path[index])
        )
    ]
    effective_orbit_path_probability = (
        orbit_path_probability if path_edges else 0.0
    )
    draw = random.random()

    if draw < effective_orbit_path_probability:
        index = random.choice(path_edges)
        sampling_orbit = candidate_by_id[id(orbit_path[index])]
        next_orbit = candidate_by_id[id(orbit_path[index + 1])]
        return (
            sampling_orbit,
            next_orbit,
            next_orbit.parent_modal_state,
            [],
        )

    if draw < (
        effective_orbit_path_probability + modal_path_probability
    ):
        sampling_orbit, modal_target, modal_path = (
            orbital_graph.modal_graph.select_guided_orbit(candidates)
        )
        if sampling_orbit is not None:
            return sampling_orbit, None, modal_target, modal_path

    return random.choice(candidates), None, None, []


def select_tensor_guidance(
    tensor_tree,
    sampling_orbit,
    incumbent_orbit_path,
    optimistic_modal_target,
    modal_graph,
    *,
    incumbent_probability=0.6,
    sampling_reuse_probability=0.25,
    modal_target_probability=0.8,
):
    """Select a reachable tensor orbit and compatible transition guidance.

    Returns ``(O_tensor, O_next, S_next)``. ``O_next`` is present only for
    ordered incumbent-path exploitation. ``S_next`` is always associated with
    the returned ``O_tensor``; an outer-loop modal target is reused only when
    the outer sampling orbit itself is selected.
    """
    if (
        incumbent_probability < 0.0
        or sampling_reuse_probability < 0.0
        or incumbent_probability + sampling_reuse_probability > 1.0
        or not 0.0 <= modal_target_probability <= 1.0
    ):
        raise ValueError("Invalid tensor-guidance probabilities.")

    candidates = [
        orbit
        for orbit in tensor_tree.orbit_experience
        if tensor_tree.orbit_nodes.get(id(orbit))
    ]
    if not candidates:
        return None, None, None

    candidate_by_id = {id(orbit): orbit for orbit in candidates}
    incumbent_indices = [
        index
        for index in range(len(incumbent_orbit_path or ()))
        if id(incumbent_orbit_path[index]) in candidate_by_id
    ]
    effective_incumbent_probability = (
        incumbent_probability if incumbent_indices else 0.0
    )
    draw = random.random()

    if draw < effective_incumbent_probability:
        index = random.choice(incumbent_indices)
        tensor_orbit = candidate_by_id[id(incumbent_orbit_path[index])]
        if index + 1 < len(incumbent_orbit_path):
            next_orbit = incumbent_orbit_path[index + 1]
            return tensor_orbit, next_orbit, next_orbit.parent_modal_state
        # The final incumbent orbit has no outgoing transition. Selecting it
        # gives the tensor tree explicit effort to connect to its goal nodes.
        return tensor_orbit, None, None

    sampling_orbit_in_tree = candidate_by_id.get(id(sampling_orbit))
    reuse_threshold = (
        effective_incumbent_probability + sampling_reuse_probability
    )
    if draw < reuse_threshold and sampling_orbit_in_tree is not None:
        return sampling_orbit_in_tree, None, optimistic_modal_target

    tensor_orbit = random.choice(candidates)
    desired_modal_state = None
    if random.random() < modal_target_probability:
        desired_modal_state = modal_graph.optimistic_modal_target(
            tensor_orbit.parent_modal_state
        )
    return tensor_orbit, None, desired_modal_state


def tensor_path_frontier_index(tensor_tree, orbit_path):
    """Return the source index of the deepest incomplete path edge.

    An edge is complete only when the tensor tree contains a complete
    transition node shared by both adjacent orbits. Merely finding tensor
    nodes in the successor orbit is insufficient because it may have been
    reached through another orbit sequence. If every edge is complete, the
    final orbit index is returned so expansion can connect to its goal nodes.
    """
    incomplete_index = tensor_path_incomplete_frontier_index(
        tensor_tree,
        orbit_path,
    )
    if incomplete_index is not None:
        return incomplete_index
    if not orbit_path:
        return None

    final_index = len(orbit_path) - 1
    final_orbit = orbit_path[final_index]
    if tensor_tree.orbit_nodes.get(id(final_orbit)):
        return final_index
    return None


def tensor_path_incomplete_frontier_index(tensor_tree, orbit_path):
    """Return the first incomplete edge source, or ``None`` if complete."""
    if not orbit_path:
        return None

    for index, (source_orbit, next_orbit) in enumerate(
        zip(orbit_path, orbit_path[1:])
    ):
        source_keys = tensor_tree.orbit_nodes.get(id(source_orbit), set())
        if not source_keys:
            return None
        next_keys = tensor_tree.orbit_nodes.get(id(next_orbit), set())
        shared_keys = source_keys & next_keys
        edge_is_complete = any(
            key in tensor_tree.nodes
            and tensor_tree.is_transition_configuration(
                tensor_tree.nodes[key]
            )
            and getattr(
                tensor_tree.nodes[key],
                "parent_edge_orbit",
                source_orbit,
            ) is source_orbit
            and source_orbit in tensor_tree.nodes[key].parent_orbit
            and next_orbit in tensor_tree.nodes[key].parent_orbit
            for key in shared_keys
        )
        if not edge_is_complete:
            return index
    return None


def select_pre_solution_frontier_guidance(
    tensor_tree,
    orbit_path,
    *,
    frontier_probability=0.8,
):
    """Focus on the deepest incomplete path edge before the first solution.

    The complementary probability samples every root-reachable tensor orbit,
    so an incorrect provisional orbit path cannot make the search incomplete.
    """
    if not 0.0 <= frontier_probability <= 1.0:
        raise ValueError("frontier_probability must be in [0, 1].")

    candidates = [
        orbit
        for orbit in tensor_tree.orbit_experience
        if tensor_tree.orbit_nodes.get(id(orbit))
    ]
    if not candidates:
        return None, None, None

    candidate_by_id = {id(orbit): orbit for orbit in candidates}
    frontier_index = tensor_path_frontier_index(tensor_tree, orbit_path)
    frontier_orbit = (
        orbit_path[frontier_index]
        if frontier_index is not None
        else None
    )
    frontier_orbit = candidate_by_id.get(id(frontier_orbit))

    if frontier_orbit is not None and random.random() < frontier_probability:
        next_orbit = (
            orbit_path[frontier_index + 1]
            if frontier_index + 1 < len(orbit_path)
            else None
        )
        desired_state = (
            next_orbit.parent_modal_state
            if next_orbit is not None
            else None
        )
        return frontier_orbit, next_orbit, desired_state

    return random.choice(candidates), None, None



def _incumbent_adjacent_postures(
    tensor_tree,
    source_orbit,
    next_orbit,
):
    """Find waiting-posture candidates around an incumbent orbit transition.

    Args:
        tensor_tree: Tensor-product tree containing the current ``best_path``.
        source_orbit: Orbit in which the incumbent enters the transition.
        next_orbit: Orbit in which the incumbent leaves the transition.

    Returns:
        A list containing zero or two composite robot postures. The function
        only reads the incumbent path and does not modify its tensor nodes.
    """
    path = tensor_tree.best_path
    if not path or next_orbit is None:
        return []

    postures = []
    for index in range(1, len(path) - 1):
        transition_node = path[index]
        outgoing_node = path[index + 1]
        if not tensor_tree.is_transition_configuration(transition_node):
            continue
        if transition_node.parent_edge_orbit is not source_orbit:
            continue
        if outgoing_node.parent_edge_orbit is not next_orbit:
            continue

        postures.append(tensor_tree.motion_joint_angles(path[index - 1]))
        postures.append(tensor_tree.motion_joint_angles(outgoing_node))
        break
    return postures


def generate_incumbent_waiting_posture_variants(
    boundary_configuration,
    source_orbit,
    next_orbit,
    tensor_tree,
    environment,
    max_variants=2,
):
    """Create alternative boundaries using useful incumbent waiting postures.

    Args:
        boundary_configuration: Original valid composite boundary sample.
        source_orbit: Orbit from which the sampled action is executed.
        next_orbit: Canonical neighboring orbit produced by that action.
        tensor_tree: Tensor tree holding the current incumbent solution.
        environment: Planning environment used for composite boundary checks.
        max_variants: Maximum number of valid alternatives returned for this
            original boundary.

    Returns:
        A list of at most ``max_variants`` validated composite transition
        configurations. The original boundary is not included in this list.
    """
    if max_variants <= 0 or not hasattr(environment, "is_feasible_at_boundary"):
        return []

    participating_robots = {
        robot_id
        for robot_id in source_orbit.attachments
        if source_orbit.attachments.get(robot_id)
        != boundary_configuration.attachments.get(robot_id)
    }
    if not participating_robots:
        return []
    waiting_robots = [
        robot_id
        for robot_id in sorted(source_orbit.attachments)
        if robot_id not in participating_robots
    ]
    if not waiting_robots:
        return []

    candidate_postures = _incumbent_adjacent_postures(
        tensor_tree,
        source_orbit,
        next_orbit,
    )
    variants = []
    seen_waiting_postures = set()
    for posture in candidate_postures:
        if len(posture) != len(boundary_configuration.joint_angles):
            continue

        waiting_key = tuple(
            tuple(np.round(np.asarray(posture[robot_id], dtype=float), 5))
            for robot_id in waiting_robots
        )
        if waiting_key in seen_waiting_postures:
            continue
        seen_waiting_postures.add(waiting_key)

        joint_angles = copy.deepcopy(boundary_configuration.joint_angles)
        pre_pose = copy.deepcopy(boundary_configuration.pre_pose)
        changed = False
        for robot_id in waiting_robots:
            candidate = copy.deepcopy(posture[robot_id])
            if not np.allclose(joint_angles[robot_id], candidate):
                changed = True
            joint_angles[robot_id] = candidate
            pre_pose[robot_id] = copy.deepcopy(candidate)
        if not changed:
            continue

        variant = CompositeConfiguration(
            joint_angles=joint_angles,
            pre_pose=pre_pose,
            attachments=copy.deepcopy(boundary_configuration.attachments),
            grasp_poses=copy.deepcopy(boundary_configuration.grasp_poses),
            placement_poses=copy.deepcopy(
                boundary_configuration.placement_poses
            ),
            is_transition=True,
            parent_orbit=source_orbit,
        )
        if environment.is_feasible_at_boundary(variant, source_orbit):
            variants.append(variant)
            if len(variants) >= max_variants:
                break
    return variants


def algo(N_m, N_t, env, env_name,
         tensor_expansion_per_iteration=50, max_iter=100, print_frequency=10, 
         show_final_path=True, random_seed=None, show_cost_plot=True,
         lazy_level1=False, incumbent_waiting_postures=False,
         max_incumbent_waiting_variants=2,
         pre_goal_tensor_expansion=10,
         post_goal_tensor_expansion=50,
         max_planning_time=None,
         return_solution_statistics=False):

    '''
        N_m: number of sampling iterior points
        N_t: number of sampling boundary points
        lazy_level1: defer constituent motion checks until candidate solutions.
        incumbent_waiting_postures: add alternative complete transitions whose
            nonparticipating robots reuse postures adjacent to the matching
            transition in the current incumbent tensor path.
        pre_goal_tensor_expansion: tensor-tree expansion budget used until a
            goal orbit has been discovered. Keeping this small prevents the
            tensor tree from growing rapidly while the modal/orbit graph is
            still searching for a task-complete orbit.
        post_goal_tensor_expansion: tensor-tree expansion budget used after a
            goal orbit has been discovered.
        max_planning_time: optional soft wall-clock limit in seconds. The
            planner stops at this limit even if ``max_iter`` was not reached.
        return_solution_statistics: when true, return a fourth value containing
            first-solution and convergence measurements for experiment logs.
    '''

    if pre_goal_tensor_expansion < 1 or post_goal_tensor_expansion < 1:
        raise ValueError(
            "Tensor expansion budgets must both be positive integers."
        )
    if max_planning_time is not None:
        max_planning_time = float(max_planning_time)
        if not np.isfinite(max_planning_time) or max_planning_time <= 0.0:
            raise ValueError("max_planning_time must be a positive number.")

    planner_start_time = time.perf_counter()
    planner_deadline = (
        planner_start_time + max_planning_time
        if max_planning_time is not None
        else None
    )

    # Timing profiling
    time_stats = {
        # Timing (seconds)
        "nodes_sampling": 0.0,
        "consituent_roadmap_construction": 0.0,
        "dijkstra": 0.0,
        "tensor_product_tree_construction": 0.0
    }
    _set_global_seed(random_seed)
    _configure_deterministic_pybullet(env)


    # Init q_s
    init_joint_angles = env.get_init_configuration()
    q_s = CompositeConfiguration(
        joint_angles=init_joint_angles, 
        is_transition=False, 
        attachments=env.init_attachments
    )
    
    # Init orbital graph with O_S
    init_orbit_states = {
        'robot_states': env.init_robot_states,
        'object_states': env.init_object_states,
        'attachments': env.init_attachments,
        'grasp_poses': env.init_grasp_poses,
        'placement_poses': env.init_placement_poses
    }
    orbital_graph = OrbitGraph(q_s, init_orbit_states, env)
    orbit_S = q_s.parent_orbit[0]  # initial orbit, q_s should have only one parent orbit

    # Init constituent roadmap graph with the component of q_s
    constituent_roadmaps = {}
    for robot in env.robots:
        robot_id = robot.id
        robot_configuration = q_s.joint_angles[robot_id]
        robot_orbit = orbit_S                        
        con_roadmap = ConstituentRoadmap(
            robot_id=robot_id, 
            init_joint_angles=robot_configuration, 
            init_orbit=robot_orbit, 
            environment=env,
            lazy_level1=lazy_level1)
        constituent_roadmaps[robot_id] = con_roadmap

    # Init the search tree for drrt* with V_s = (O_S, q_s)
    root_constituent_nodes = []
    for robot_id, con_roadmap in constituent_roadmaps.items():
        robot_root_node = con_roadmap.root_node
        root_constituent_nodes.append(robot_root_node)
    root_tensor_node = TensorProductNode(constituent_nodes=root_constituent_nodes, parent_orbit=orbit_S)
    tensor_tree = TensorProductTree(root_tensor_node=root_tensor_node, root_orbit=orbit_S, environment=env)



    # To promote exploitation in later phase of algorithm
    incumbent_orbit_path = []           # Best known symbolic/orbit-level path.
    modal_orbit_bias = 0.6              # Probability of modal-state-guided outer sampling before an orbit path exists.
    orbit_path_sampling_bias = 0.5      # Probability of sampling an edge from the incumbent orbit path when one exists.
    modal_orbit_bias_with_orbit_path = 0.3      # Probability of modal-state-guided outer sampling when an incumbent orbit path exists.
    modal_transition_bias = 0.4         # Probability of accepting a transition matching the requested next orbit/modal target.
    tensor_incumbent_probability = 0.6  # Probability of tensor expansion on the incumbent path after a valid tensor solution exists.
    tensor_frontier_probability_pre_solution = 0.5      # Probability of tensor expansion on the deepest incomplete incumbent-path frontier before a valid solution exists.
    tensor_reuse_probability_pre_solution = 0.5         # Probability of reusing the outer-sampled orbit in tensor expansion before a solution exists, when frontier guidance is inactive.
    tensor_reuse_probability_post_solution = 0.25       # Probability of reusing the outer-sampled orbit in tensor expansion after a valid solution exists.
    tensor_modal_target_probability = 0.6               # Probability of assigning a modal target during otherwise unguided tensor-orbit selection.

    
    # Cofig for stacking 4 objects
    # modal_orbit_bias = 0.7
    # orbit_path_sampling_bias = 0.6
    # modal_orbit_bias_with_orbit_path = 0.3
    # modal_transition_bias = 0.7
    # tensor_incumbent_probability = 0.6
    # tensor_frontier_probability_pre_solution = 0.8
    # tensor_reuse_probability_pre_solution = 0.6
    # tensor_reuse_probability_post_solution = 0.25
    # tensor_modal_target_probability = 0.8


    ### Test values for 4 robots pick and place
    # Outer search: keep exploring alternative task/orbit routes
    # modal_orbit_bias = 0.3
    # orbit_path_sampling_bias = 0.4
    # modal_orbit_bias_with_orbit_path = 0.2
    # modal_transition_bias = 0.3

    # # Before the first valid tensor solution: help complete the current route,
    # # but retain chances to explore alternatives
    # tensor_frontier_probability_pre_solution = 0.5
    # tensor_reuse_probability_pre_solution = 0.4

    # # After a valid solution: spend more expansions refining it, while
    # # retaining some exploration
    # tensor_incumbent_probability = 0.6
    # tensor_reuse_probability_post_solution = 0.25
    # tensor_modal_target_probability = 0.4




    # For plotting 
    cost_history = []
    convergence_history = []
    first_solution_iteration = None
    first_solution_time = None
    first_solution_cost = None
    previous_print_orbit_count = len(orbital_graph.orbit_graph)
    previous_print_rewire_count = 0
    previous_print_time = time.perf_counter()
    previous_print_time_stats = dict(time_stats)
    incumbent_waiting_variants_added = 0
    completed_iterations = 0
    stop_reason = "max_iterations"

    try:
        for iter in range(max_iter):
            if (
                planner_deadline is not None
                and time.perf_counter() >= planner_deadline
            ):
                stop_reason = "time_limit"
                print(
                    "Planning time limit reached before iteration "
                    f"{iter}."
                )
                break
            if iter % print_frequency == 0:
                print(f"Iteration {iter}....")

            desired_modal_state = None
            preferred_modal_path = []
            sampling_next_orbit = None

            # Keep the optimistic modal path, but prefer ordered realized
            # orbit-path edges once either a provisional orbital solution or a
            # validated tensor solution has supplied such a path.
            (
                selected_orbit,
                sampling_next_orbit,
                desired_modal_state,
                preferred_modal_path,
            ) = select_sampling_guidance(
                orbital_graph,
                incumbent_orbit_path,
                orbit_path_probability=(
                    orbit_path_sampling_bias if incumbent_orbit_path else 0.0
                ),
                modal_path_probability=(
                    modal_orbit_bias_with_orbit_path
                    if incumbent_orbit_path
                    else modal_orbit_bias
                ),
            )
            if selected_orbit is None:
                raise RuntimeError("The orbital graph contains no orbit.")

            # Sample interior configurations
            t0 = time.perf_counter()
            interior_configurations = selected_orbit.sample_interior_points(N_m, env)
            time_stats["nodes_sampling"] += time.perf_counter() - t0


            # Add the sampled interior configurations to the constituent roadmap graphs 
            # and connect them to their nearest neighbors
            t0 = time.perf_counter()
            for point in interior_configurations:
                for robot_id, con_roadmap in constituent_roadmaps.items():
                    robot_configuration = point.joint_angles[robot_id]
                    robot_orbit = point.parent_orbit[0]                         # interior configuration should have single parent orbit
                    robot_attachment = point.attachments[robot_id] if point.attachments else None
                    con_roadmap.add_interior_node(robot_configuration, robot_orbit, attachment=robot_attachment)
            time_stats["consituent_roadmap_construction"] += time.perf_counter() - t0


            # Sample boundary configurations
            t0 = time.perf_counter()
            boundary_configurations = selected_orbit.sample_boundary_points(
                N_t,
                env,
                desired_next_orbit=sampling_next_orbit,
                desired_modal_state=desired_modal_state,
                modal_graph=orbital_graph.modal_graph,
                modal_transition_bias=modal_transition_bias,
            )
            time_stats["nodes_sampling"] += time.perf_counter() - t0


            # Add boundary configurations to constituent roadmap graphs, connect with neighbors 
            # and update orbital graph
            for boundary_point in boundary_configurations:

                # update orbit graph
                neighboring_orbit = selected_orbit.define_neighboring_orbit(boundary_point)
                orbit_in_graph = orbital_graph.add_node(neighboring_orbit)
                orbital_graph.add_edge(selected_orbit, orbit_in_graph)

                transition_configurations = [boundary_point]
                if incumbent_waiting_postures:
                    t0 = time.perf_counter()
                    waiting_variants = (
                        generate_incumbent_waiting_posture_variants(
                            boundary_configuration=boundary_point,
                            source_orbit=selected_orbit,
                            next_orbit=orbit_in_graph,
                            tensor_tree=tensor_tree,
                            environment=env,
                            max_variants=max_incumbent_waiting_variants,
                        )
                    )
                    time_stats["nodes_sampling"] += time.perf_counter() - t0
                    transition_configurations.extend(waiting_variants)
                    incumbent_waiting_variants_added += len(waiting_variants)

                # Add the original boundary and each valid waiting-posture
                # alternative as separate synchronized transition keys.
                t0 = time.perf_counter()
                for transition_configuration in transition_configurations:
                    for robot_id, con_roadmap in constituent_roadmaps.items():
                        robot_configuration = (
                            transition_configuration.joint_angles[robot_id]
                        )
                        robot_attachment = (
                            transition_configuration.attachments[robot_id]
                            if transition_configuration.attachments
                            else None
                        )
                        previously_attached_object = (
                            selected_orbit.attachments.get(robot_id)
                        )
                        robot_grasp_pose = (
                            transition_configuration.grasp_poses.get(
                                robot_attachment
                            )
                        )
                        robot_placement_pose = (
                            transition_configuration.placement_poses.get(
                                previously_attached_object
                            )
                        )
                        robot_pre_pose = (
                            transition_configuration.pre_pose[robot_id]
                        )
                        con_roadmap.add_boundary_node(
                            robot_configuration,
                            [selected_orbit, orbit_in_graph],
                            transition_key=transition_configuration,
                            grasp_pose=robot_grasp_pose,
                            placement_pose=robot_placement_pose,
                            pre_pose=robot_pre_pose,
                        )
                time_stats["consituent_roadmap_construction"] += time.perf_counter() - t0


            # Goal configurations can only be inserted after a goal orbit has
            # been discovered, but the tensor tree should grow before then.
            goal_orbit_reachable = orbital_graph.is_goal_orbit_reachable()
            if not goal_orbit_reachable:
                if iter % print_frequency == 0:
                    print(
                        "Goal orbit is not reachable from the initial orbit. "
                        "Expanding the tensor tree without goal guidance..."
                    )
            else:
                goal_robot_poses, goal_object_poses, goal_attachments = env._get_goal_conditions()
                goal_orbits  = orbital_graph.get_goal_orbits()
                goal_configs = []
                if goal_orbits:
                    for goal_orbit in goal_orbits:
                        goal_config = CompositeConfiguration(
                                            joint_angles=goal_robot_poses,
                                            is_transition=False,
                                            parent_orbit=[goal_orbit],
                                            attachments=goal_attachments
                                        )
                        goal_config = goal_orbit.add_goal_configuration(goal_config)
                        goal_configs.append(goal_config)
                # add goal configuration to the constituent roadmap graphs and connect with neighbors
                for goal_config in goal_configs:
                    for robot_id, con_roadmap in constituent_roadmaps.items():
                        robot_configuration = goal_config.joint_angles[robot_id]
                        robot_orbit = goal_config.parent_orbit[0]  # FIXME: assume that goal configuration should have single parent orbit
                        con_roadmap.add_goal_node(robot_configuration, robot_orbit)

            # Roadmap topology changes every iteration. Refresh Algorithm 7
            # before transition ranking and tensor-tree expansion.
            t0 = time.perf_counter()
            orbital_graph.update_heuristic_values()
            for con_roadmap in constituent_roadmaps.values():
                con_roadmap.update_heuristic_values()
            time_stats["dijkstra"] += time.perf_counter() - t0

            has_tensor_solution = bool(tensor_tree.best_path)
            pre_solution_frontier_active = bool(
                incumbent_orbit_path and not has_tensor_solution
            )
            if pre_solution_frontier_active:
                tensor_orbit, tensor_next_orbit, tensor_modal_target = (
                    select_pre_solution_frontier_guidance(
                        tensor_tree,
                        incumbent_orbit_path,
                        frontier_probability=(
                            tensor_frontier_probability_pre_solution
                        ),
                    )
                )
            else:
                tensor_reuse_probability = (
                    tensor_reuse_probability_post_solution
                    if has_tensor_solution
                    else tensor_reuse_probability_pre_solution
                )
                tensor_orbit, tensor_next_orbit, tensor_modal_target = (
                    select_tensor_guidance(
                        tensor_tree=tensor_tree,
                        sampling_orbit=selected_orbit,
                        incumbent_orbit_path=incumbent_orbit_path,
                        optimistic_modal_target=desired_modal_state,
                        modal_graph=orbital_graph.modal_graph,
                        incumbent_probability=(
                            tensor_incumbent_probability
                            if has_tensor_solution
                            else 0.0
                        ),
                        sampling_reuse_probability=tensor_reuse_probability,
                        modal_target_probability=tensor_modal_target_probability,
                    )
                )
            if tensor_orbit is None:
                raise RuntimeError(
                    "The tensor tree contains no reachable expansion orbit."
                )

            t0  = time.perf_counter()
            best_tensor_path, best_cost = tensor_tree.expand_tree(
                constituent_roadmaps,
                max_iterations=tensor_expansion_per_iteration,
                preferred_orbit=tensor_orbit,
                preferred_next_orbit=tensor_next_orbit,
                desired_modal_state=tensor_modal_target,
                modal_graph=orbital_graph.modal_graph,
                modal_transition_bias=modal_transition_bias,
                deadline=planner_deadline,
            )
            time_stats["tensor_product_tree_construction"] += time.perf_counter() - t0
            best_path, best_cost = tensor_tree.to_composite_path(best_tensor_path)
            has_finite_solution = bool(
                best_path
                and best_cost is not None
                and np.isfinite(float(best_cost))
            )
            if has_finite_solution:
                numeric_cost = float(best_cost)
                if first_solution_time is None:
                    first_solution_iteration = iter
                    first_solution_time = (
                        time.perf_counter() - planner_start_time
                    )
                    first_solution_cost = numeric_cost
            if iter % print_frequency == 0:
                recorded_cost = (
                    float(best_cost) if has_finite_solution else None
                )
                cost_history.append((iter, recorded_cost))
                convergence_history.append({
                    "iteration": iter,
                    "elapsed_time": time.perf_counter() - planner_start_time,
                    "cost": recorded_cost,
                })
            best_tensor_orbit_path = (
                get_orbit_path_from_path(best_tensor_path)
                if best_tensor_path
                else []
            )

            if best_tensor_orbit_path:
                incumbent_orbit_path = list(best_tensor_orbit_path)
            elif orbital_graph.has_found_goal_orbit():
                provisional_orbit_path = (
                    orbital_graph.best_goal_orbit_path()
                )
                if provisional_orbit_path:
                    incumbent_orbit_path = provisional_orbit_path

            if iter % print_frequency == 0:
                current_print_time = time.perf_counter()
                interval_runtime = current_print_time - previous_print_time
                interval_time_stats = {
                    key: value - previous_print_time_stats[key]
                    for key, value in time_stats.items()
                }
                information = tensor_tree._get_information()
                current_orbit_count = len(orbital_graph.orbit_graph)
                current_rewire_count = information["successful_rewires"]
                roadmap_sizes = {
                    robot_id: len(roadmap.graph)
                    for robot_id, roadmap in constituent_roadmaps.items()
                }
                print(
                    "Planner diagnostics: "
                    f"C_best={best_cost:.4f} \n"
                    f"N_orbit={current_orbit_count}, "
                    f"N_new_orbit=+{current_orbit_count - previous_print_orbit_count}, "
                    f"N_mode={len(orbital_graph.modal_graph.modes_by_key)}, "
                    f"N_modal_state={len(orbital_graph.modal_graph)} \n"
                    "has_found_goal_modal_state="
                    f"{orbital_graph.has_found_goal_modal_state()}, "
                    f"has_found_goal_orbit={orbital_graph.has_found_goal_orbit()}, "
                    "is_modal_goal_state_reachable="
                    f"{orbital_graph.modal_graph.is_goal_state_reachable()}, "
                    "orbit_path_guidance_active="
                    f"{sampling_next_orbit is not None}, "
                    "orbit_guidance_path_length="
                    f"{len(incumbent_orbit_path)}, "
                    "pre_solution_frontier_active="
                    f"{pre_solution_frontier_active}, "
                    "modal_guidance_active="
                    f"{sampling_next_orbit is None and desired_modal_state is not None}, "
                    f"preferred_modal_path_length={len(preferred_modal_path)} \n"
                    f"|G_i|={roadmap_sizes}, "
                    f"|T|={information['num_nodes']}, "
                    f"Tensor_tree_explored_orbit={information['num_explored_orbits']} \n"
                    f"N_rewire={current_rewire_count} "
                    f"(+{current_rewire_count - previous_print_rewire_count}), "
                    f"N_lazy_level1_checks={information['lazy_level1_checks']}, "
                    f"N_invalid_level1_contexts={information['invalid_level1_contexts']}, "
                    "N_incumbent_waiting_variants="
                    f"{incumbent_waiting_variants_added}, "
                    "N_target_dijkstra="
                    f"{information['target_distance_searches']}, "
                    "N_target_cache_hit="
                    f"{information['target_distance_cache_hits']} \n"
                    "N_lazy_tensor_checks="
                    f"{information['lazy_edge_validation_checks']}, "
                    "N_invalid_tensor_edges="
                    f"{information['known_invalid_tensor_edges']}, "
                    f"interval_runtime={interval_runtime:.4f}s, "
                )
                previous_print_time = current_print_time
                previous_print_time_stats = dict(time_stats)
                previous_print_orbit_count = current_orbit_count
                previous_print_rewire_count = current_rewire_count
                if best_path:
                    print(f"Path from initial configuration to Goal has {len(best_path)} nodes")
                    orbit_number_by_id = {
                        id(orbit): index
                        for index, orbit in enumerate(
                            orbital_graph.orbit_graph,
                            start=1,
                        )
                    }
                    orbit_sequence = " -> ".join(
                        f"O{orbit_number_by_id.get(id(orbit), '?')}"
                        for orbit in best_tensor_orbit_path
                    )
                    print(
                        "Best tensor path orbit sequence "
                        f"({len(best_tensor_orbit_path)} orbits): "
                        f"{orbit_sequence}"
                    )
                else:
                    print("No path found from initial configuration to Goal.")
                    # print("Check orbit list:", [orb.object_states for orb in orbital_graph.graph])
                if best_path:
                    path_filename = (
                        f"{env_name}_planned_path_seed_{random_seed}_{alg_name}.txt"
                        if random_seed is not None
                        else f"{env_name}_planned_path_{alg_name}.txt"
                    )
                    save_path_to_file(best_path, path_cost=best_cost, filename=path_filename, random_seed=random_seed)

                # These are deltas since the previous periodic report. The
                # final report below remains cumulative for the full trial.
                print("\nInterval timing statistics (in seconds):")
                for key, value in interval_time_stats.items():
                    print(f"{key}: {value:.4f}")
                print(
                    "tracked_interval_total: "
                    f"{sum(interval_time_stats.values()):.4f}"
                )
                print("--------------------------------------------------")
            completed_iterations = iter + 1
            if (
                planner_deadline is not None
                and time.perf_counter() >= planner_deadline
            ):
                stop_reason = "time_limit"
                print(
                    "Planning time limit reached after completing iteration "
                    f"{iter}."
                )
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("KeyboardInterrupt detected. Stopping the algorithm.")
            
    finally:
        best_tensor_path = tensor_tree.best_path
        best_path, best_cost = tensor_tree.to_composite_path(best_tensor_path)
        print("Final path from initial configuration to Goal:")
        if best_path:
            print(f"Path from initial configuration to Goal has {len(best_path)} nodes. Cost: {best_cost:.4f}")
            path_filename = (
                f"{env_name}_planned_path_seed_{random_seed}_{alg_name}.txt"
                if random_seed is not None
                else f"{env_name}_planned_path_{alg_name}.txt"
            )
            save_path_to_file(
                best_path,
                path_cost=best_cost,
                filename=path_filename,
                random_seed=random_seed,
            )
        else:
            print("No path found from initial configuration to Goal.")

        if show_final_path and best_path:
            sim_filename = (
                f"{env_name}_path_simulation_{random_seed}_{alg_name}.mp4"
                if random_seed is not None
                else f"{env_name}_path_simulation_{alg_name}.mp4"
            )
            env.simulate_final_path(path_filename, output_sim_filename=sim_filename, delay=0.2)

        if show_cost_plot:
            plot_path_cost(cost_history, filename=f"{env_name}_convergence_{random_seed}_{alg_name}.png")

        save_orbit_graph_to_file(orbital_graph, filename=f"{env_name}_orbit_graph_{random_seed}_{alg_name}.txt")

        # Report timing statistics
        print("\nTiming statistics (in seconds):")
        for key, value in time_stats.items():
            print(f"{key}: {value:.4f}")
        print(f"Stop reason: {stop_reason}")
        print(f"Completed planner iterations: {completed_iterations}")

    final_solution_cost = (
        float(best_cost)
        if best_path
        and best_cost is not None
        and np.isfinite(float(best_cost))
        else None
    )
    solution_statistics = {
        "stop_reason": stop_reason,
        "completed_iterations": completed_iterations,
        "max_planning_time": max_planning_time,
        "first_solution_iteration": first_solution_iteration,
        "first_solution_time": first_solution_time,
        "first_solution_cost": first_solution_cost,
        "final_solution_cost": final_solution_cost,
        "convergence": convergence_history,
    }
    if return_solution_statistics:
        return best_path, best_cost, time_stats, solution_statistics
    return best_path, best_cost, time_stats



if __name__ == "__main__":
    debugger_attached = sys.gettrace() is not None

    # Add/remove environments here. They will run sequentially in this order.
    environment_factories = [
        # ("2r_pick_place_2_objects", lambda: SimplePickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("2r_pick_place_4_objects", lambda: SimplePickPlaceEnv(num_robots=2, num_mobjects=4, show_gui=False)),
        # ("2r_pick_place_6_objects", lambda: SimplePickPlaceEnv(num_robots=2, num_mobjects=6, show_gui=False)),
        # ("3_robots_pick_place_3_objs_lazy", lambda: ThreeRobotsPickPlaceEnv(num_robots=3, num_mobjects=3, show_gui=False)),
        # ("small_box_pick_place", lambda: SmallBoxPickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("shelf_pick_place", lambda: ShelfPickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("new_handover", lambda: HandoverEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("stacking_4_objects", lambda: StackingObjectsEnv(num_robots=2, num_mobjects=4, show_gui=False)),
        ("stacking_castle", lambda: StackingCastleEnv(num_robots=2, num_mobjects=4, show_gui=False)),
        # ("4robots_pick_place_4objs", lambda: FourRobotPickPlaceEnv(num_robots=4, num_mobjects=4, show_gui=False)),
        
        # ("simple_motion_planning_2objs", lambda: SimpleMotionPlanningWithSubgoalEnv(num_robots=2, num_obstacles=4, num_mobjects=2)),
    ]


    ### HYPERPARAMETERS ###

    # Enable optimistic constituent edges; Level-2 checking is always lazy.
    lazy_level1 = True

    trial = 2
    base_random_seed = 1
    num_interior_samples = 30
    num_transition_samples = 50
    max_planner_iterations = 500    
    max_planner_time_seconds = 600 # Set to a positive number of seconds to enable the soft time limit.
    diagnostics_frequency = 10
    use_incumbent_waiting_postures = True
    max_incumbent_waiting_variants = 10


    ### ALGORITHM EXECUTION ###

    for env_idx, (env_name, env_factory) in enumerate(environment_factories, start=1):
        print("=" * 80)
        print(f"Environment {env_idx}/{len(environment_factories)}: {env_name}")
        print("=" * 80)

        _cleanup_between_environments()
        _set_global_seed(base_random_seed)
        env = env_factory()

        runtime = []
        distances = []
        trial_results = []
        random_seed = base_random_seed
        statistics_filename = (
            f"{env_name}_trial_statistics_seeds_"
            f"{base_random_seed}-{base_random_seed + trial - 1}_{alg_name}.txt"
        )
        statistics_metadata = {
            "Environment": env_name,
            "Environment class": type(env).__name__,
            "Algorithm": alg_name,
            "Requested trials": trial,
            "Base random seed": base_random_seed,
            "Lazy Level-1 collision checking": lazy_level1,
            "Interior samples per iteration": num_interior_samples,
            "Transition samples per iteration": num_transition_samples,
            "Maximum planner iterations": max_planner_iterations,
            "Maximum planning time (seconds)": max_planner_time_seconds,
            "Diagnostics frequency": diagnostics_frequency,
            "Incumbent waiting postures": use_incumbent_waiting_postures,
            "Maximum incumbent waiting variants per boundary": (
                max_incumbent_waiting_variants
            ),
        }

        try:
            for i in range(trial):
                print(f"Trial {i+1}/{trial}")
                print(f"Using random seed: {random_seed}")
                print(f"Lazy level1 collision checking: {lazy_level1}")
                _set_global_seed(random_seed)
                env.reset()
                start = time.perf_counter()
                try:
                    path, distance, time_stats, solution_statistics = algo(
                        N_m=num_interior_samples,
                        N_t=num_transition_samples,
                        tensor_expansion_per_iteration=50,
                        max_iter=max_planner_iterations,
                        max_planning_time=max_planner_time_seconds,
                        print_frequency=diagnostics_frequency,
                        show_final_path=False,
                        env=env,
                        env_name=env_name,
                        random_seed=random_seed,
                        show_cost_plot=True,
                        lazy_level1=lazy_level1,
                        incumbent_waiting_postures=(
                            use_incumbent_waiting_postures
                        ),
                        max_incumbent_waiting_variants=(
                            max_incumbent_waiting_variants
                        ),
                        return_solution_statistics=True,
                    )
                except Exception as error:
                    trial_runtime = time.perf_counter() - start
                    trial_results.append({
                        "seed": random_seed,
                        "success": False,
                        "runtime": trial_runtime,
                        "best_cost": float("inf"),
                        "path_length": 0,
                        "time_stats": {},
                        "solution_statistics": {
                            "stop_reason": "error",
                            "completed_iterations": 0,
                            "max_planning_time": max_planner_time_seconds,
                            "first_solution_iteration": None,
                            "first_solution_time": None,
                            "first_solution_cost": None,
                            "final_solution_cost": None,
                            "convergence": [],
                        },
                        "error": f"{type(error).__name__}: {error}",
                    })
                    save_trial_statistics(
                        trial_results,
                        statistics_filename,
                        metadata=statistics_metadata,
                    )
                    raise

                trial_runtime = time.perf_counter() - start
                print(
                    f"Trial {i+1} running time: "
                    f"{trial_runtime:.2f} seconds."
                    f"\n=========== END OF TRIAL {i+1} ===========\n"
                )
                runtime.append(trial_runtime)
                distances.append(distance if distance is not None else float('inf'))
                trial_succeeded = (
                    bool(path)
                    and distance is not None
                    and np.isfinite(float(distance))
                )
                trial_results.append({
                    "seed": random_seed,
                    "success": trial_succeeded,
                    "runtime": trial_runtime,
                    "best_cost": distance,
                    "path_length": len(path) if path else 0,
                    "time_stats": dict(time_stats),
                    "solution_statistics": solution_statistics,
                })
                save_trial_statistics(
                    trial_results,
                    statistics_filename,
                    metadata=statistics_metadata,
                )
                random_seed += 1

            if debugger_attached and p.isConnected(env.visualization_client):
                env.keep_gui_alive()

            print(f"Average runtime over {trial} trials: {np.mean(runtime):.2f} seconds; std: {np.std(runtime):.2f} seconds.")
            print(f"Average path cost over {trial} trials: {np.mean(distances):.4f}; std: {np.std(distances):.4f}.")
        finally:
            if not debugger_attached:
                env.close()
            else:
                print("Debugger mode: skipping automatic env.close().")

            del env
            _cleanup_between_environments()
