"""Unbiased constituent-roadmap dRRT* ablation.

This entry point intentionally leaves ``drrt_star_main.py`` untouched.  It
reuses its roadmap construction, transition sampler, tensor-product tree,
collision checking, stopping criteria, and cost function, while replacing
only the three high-level guidance selectors with uniform exploration:

* outer orbit selection;
* tensor-tree orbit/frontier selection; and
* modal/ordered transition targeting.

The public ``algo`` signature follows ``drrt_star_main.algo``.  Level-1
constituent collision checking is lazy by default for this baseline.
"""

import time
import numpy as np

from planners import drrt_star_main as _planner
from planners.drrt_star_main import (
    CompositeConfiguration,
    OrbitGraph,
    ConstituentRoadmap,
    TensorProductNode,
    TensorProductTree,
    _configure_deterministic_pybullet,
    _set_global_seed,
    generate_incumbent_waiting_posture_variants,
)
from planners.utils import (
    get_orbit_path_from_path,
    plot_path_cost,
    save_orbit_graph_to_file,
    save_path_to_file,
    save_trial_statistics,
)


PLANNER_VARIANT = "dRRT_star_original"
alg_name = "drrt_star_original"





def algo(N_m, N_t, env, env_name,
         tensor_expansion_per_iteration= 50, max_iter=100, print_frequency=10, 
         show_final_path=True, random_seed=None, show_cost_plot=True,
         lazy_level1=True, incumbent_waiting_postures=False,
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

            # Original dRRT* baseline: sample an existing orbit uniformly.
            # This deliberately excludes orbit-path and modal-state guidance.
            selected_orbit = orbital_graph.uniform_sample_orbit()


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
                desired_next_orbit=None,
                desired_modal_state=None,
                modal_graph=orbital_graph.modal_graph,
                # No modal-state or ordered-transition guidance in the
                # original baseline.
                modal_transition_bias=0.0,
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


            t0  = time.perf_counter()
            best_tensor_path, best_cost = tensor_tree.expand_tree(
                constituent_roadmaps,
                max_iterations=50,
                preferred_orbit=None,
                preferred_next_orbit=None,
                desired_modal_state=None,
                modal_graph=None,
                modal_transition_bias=None,
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




def main():
    """Run the same configured five-seed experiment as the full planner."""
    environment_factories = [
        # ("2r_pick_place_2_objects", lambda: _planner.SimplePickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("2r_pick_place_4_objects", lambda: _planner.SimplePickPlaceEnv(num_robots=2, num_mobjects=4, show_gui=False)),
        # ("2r_pick_place_6_objects",lambda: _planner.SimplePickPlaceEnv(num_robots=2, num_mobjects=6, show_gui=False)),
        # ("3r_pick_place_6_objs", lambda: _planner.ThreeRobotsPickPlaceEnv(num_robots=3, num_mobjects=6, show_gui=False)),
        ("4robots_pick_place_6objs", lambda: _planner.FourRobotPickPlaceEnv(num_robots=4, num_mobjects=6, show_gui=False)),

        # ("small_box_pick_place", lambda: _planner.SmallBoxPickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("shelf_pick_place", lambda: _planner.ShelfPickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=True)),
        # ("handover", lambda: _planner.HandoverEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("stacking_4_objects", lambda: _planner.StackingObjectsEnv(num_robots=2, num_mobjects=4, show_gui=False)),
    ]
    trial = 5
    base_random_seed = 48
    num_interior_samples = 5
    num_transition_samples = 5
    max_planner_iterations = 1000       # max iterations
    max_planner_time_seconds = 900      # max time  
    diagnostics_frequency = 10
    use_incumbent_waiting_postures = True
    max_incumbent_waiting_variants = 2

    for env_name, env_factory in environment_factories:
        _planner._cleanup_between_environments()
        _planner._set_global_seed(base_random_seed)
        env = env_factory()
        trial_results = []
        statistics_filename = (
            f"{env_name}_trial_statistics_seeds_"
            f"{base_random_seed}-{base_random_seed + trial - 1}_"
            f"{alg_name}.txt"
        )
        statistics_metadata = {
            "Environment": env_name,
            "Environment class": type(env).__name__,
            "Algorithm": alg_name,
            "Planner variant": PLANNER_VARIANT,
            "Requested trials": trial,
            "Base random seed": base_random_seed,
            "Lazy Level-1 collision checking": True,
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
            for trial_index in range(trial):
                random_seed = base_random_seed + trial_index
                _planner._set_global_seed(random_seed)
                env.reset()
                start_time = time.perf_counter()
                path, cost, time_stats, solution_statistics = algo(
                    N_m=num_interior_samples,
                    N_t=num_transition_samples,
                    tensor_expansion_per_iteration=10,
                    max_iter=max_planner_iterations,
                    print_frequency=diagnostics_frequency,
                    show_final_path=False,
                    random_seed=random_seed,
                    show_cost_plot=True,
                    env=env,
                    env_name=env_name,
                    lazy_level1=True,
                    incumbent_waiting_postures=(
                        use_incumbent_waiting_postures
                    ),
                    max_incumbent_waiting_variants=(
                        max_incumbent_waiting_variants
                    ),
                    max_planning_time=max_planner_time_seconds,
                    return_solution_statistics=True,
                )
                trial_results.append({
                    "seed": random_seed,
                    "planner_variant": PLANNER_VARIANT,
                    "success": bool(
                        path
                        and cost is not None
                        and np.isfinite(float(cost))
                    ),
                    "runtime": time.perf_counter() - start_time,
                    "best_cost": cost,
                    "path_length": len(path) if path else 0,
                    "time_stats": dict(time_stats),
                    "solution_statistics": solution_statistics,
                })
                save_trial_statistics(
                    trial_results,
                    statistics_filename,
                    metadata=statistics_metadata,
                )
        finally:
            env.close()
            del env
            _planner._cleanup_between_environments()


if __name__ == "__main__":
    main()
