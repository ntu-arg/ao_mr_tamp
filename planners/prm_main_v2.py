# Run this in terminal before executing the code to avoid potential issues with deep recursion in path retracing:
# ulimit -s unlimited

"""
The practical version of the algorithm:
After number of exploration iterations, we stick to one orbit path only.

"""



import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
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
from env.three_robots_pick_place import ThreeRobotsPickPlaceEnv
from env.four_robots_pick_place import MultipleRobotsPickPlaceEnv as FourRobotPickPlaceEnv

from planners.prm_roadmap import OrbitGraph, RoadmapGraph
from planners.lazy_prm_roadmap import LazyRoadmapGraph
from planners.prm_roadmap_utils import Configuration
from planners.utils import plot_path_cost, save_path_to_file, \
    save_orbit_graph_to_file, get_orbit_path_from_path, \
    update_roadmap_graph_with_orbit_path, save_trial_statistics, cleanup_between_environments as _cleanup_between_environments, \
    set_global_seed as _set_global_seed

import faulthandler
faulthandler.enable()


alg_name = 'PRM_star'
planner_variant = 'PRM_star'


def algo(N_m, N_t, env, env_name,
         orbit_exploration_max_iter= 50, max_iter=100, print_frequency=100, 
         display_real_time=True, show_final_path=True, random_seed=None,
         show_cost_plot=True, lazy_collision_checking=True,
         max_planning_time=None, return_solution_statistics=False):

    '''
        N_m: number of sampling iterior points
        N_t: number of sampling boundary points
        lazy_collision_checking: register PRM* edges optimistically and
            validate them only when they occur on a candidate path.
        max_planning_time: optional soft wall-clock limit in seconds.
        return_solution_statistics: when true, return a fourth value with
            first-solution and convergence measurements for experiment logs.
    '''

    # Timing profiling
    time_stats = {
        # Timing (seconds)
        "nodes_sampling": 0.0,
        "edges_collision_checking": 0.0,
        "dijkstra": 0.0
    }

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



    _set_global_seed(random_seed)

    # env_name = env.__class__.__name__.replace("Env", "")
    # env_name = ''.join(['_' + c.lower() if c.isupper() else c for c in env_name]).lstrip('_')

    if display_real_time:
        fig = plt.figure()
        frames = []

    
    init_joint_angles = env.get_init_configuration()
    q_s = Configuration(
        joint_angles=init_joint_angles, 
        is_transition=False, 
        attachments=env.init_attachments
    )
    
    # init orbital graph
    init_orbit_states = {
        'robot_states': env.init_robot_states,
        'object_states': env.init_object_states,
        'attachments': env.init_attachments,
        'grasp_poses': env.init_grasp_poses,
        'placement_poses': env.init_placement_poses
    }
    orbital_graph = OrbitGraph(q_s, init_orbit_states, env)

    # init roadmap graph
    roadmap_graph = (
        LazyRoadmapGraph()
        if lazy_collision_checking
        else RoadmapGraph()
    )

    cost_history = []
    convergence_history = []
    first_solution_iteration = None
    first_solution_time = None
    first_solution_cost = None
    completed_iterations = 0
    stop_reason = "max_iterations"
    orbit_path = None
    path = None
    distance = None

    def record_solution(iteration, candidate_path, candidate_cost):
        """Record the first valid solution without changing planner state."""
        nonlocal first_solution_iteration
        nonlocal first_solution_time
        nonlocal first_solution_cost

        has_solution = bool(
            candidate_path
            and candidate_cost is not None
            and np.isfinite(float(candidate_cost))
        )
        if has_solution and first_solution_time is None:
            first_solution_iteration = iteration
            first_solution_time = time.perf_counter() - planner_start_time
            first_solution_cost = float(candidate_cost)
        return has_solution

    try:
        for iter in range(max_iter):
            if (
                planner_deadline is not None
                and time.perf_counter() >= planner_deadline
            ):
                stop_reason = "time_limit"
                print(f"Planning time limit reached before iteration {iter}.")
                break
            if iter % print_frequency == 0:
                print(f"Iteration {iter}....")

            # sample an orbit from orbital graph
            if not orbit_path:
                selected_orbit = orbital_graph.uniform_sample_orbit()
            else:
                selected_orbit = random.choice(orbit_path)


            # expand selected orbit
            t0 = time.perf_counter()
            interior_points = selected_orbit.sample_interior_points(N_m, env)
            time_stats["nodes_sampling"] += time.perf_counter() - t0

            # update roadmap graph from all iterior points
            t0 = time.perf_counter()
            for point in interior_points:
                roadmap_graph.add_interior_node(point, selected_orbit, environment=env)
            time_stats["edges_collision_checking"] += time.perf_counter() - t0

            # expand boundary of selected orbit
            t0 = time.perf_counter()
            boundary_points = selected_orbit.sample_boundary_points(N_t, env)
            time_stats["nodes_sampling"] += time.perf_counter() - t0

            # update orbital graph from boundary points
            for boundary_point in boundary_points:
                # update orbit graph
                neighboring_orbit = selected_orbit.define_neighboring_orbit(boundary_point)
                orbit_in_graph = orbital_graph.add_node(neighboring_orbit)
                orbital_graph.add_edge(selected_orbit, orbit_in_graph)
                # update roadmap graph
                t0 = time.perf_counter()
                roadmap_graph.add_boundary_node(
                    boundary_point,
                    [selected_orbit, orbit_in_graph],
                    environment=env,
                )
                time_stats["edges_collision_checking"] += time.perf_counter() - t0


            if iter == orbit_exploration_max_iter or \
                (iter > orbit_exploration_max_iter and not orbit_path and iter % print_frequency == 0):

                comp_time, path, distance = roadmap_graph.retrace_path(orbital_graph, q_s, env)
                time_stats["dijkstra"] += comp_time
                record_solution(iter, path, distance)
                # extract orbit path from the roadmap path
                if path:
                    orbit_path = get_orbit_path_from_path(path)
                    print(" >>>>>>>>> Switching to single orbit path to prioritize exploitation. <<<<<<<< \n")
                    update_roadmap_graph_with_orbit_path(roadmap_graph, orbit_path)
                else:   
                    print("No path found from initial configuration to Goal.")


            if display_real_time:
                env.draw(configurations=roadmap_graph.graph.keys())
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()

            if iter % print_frequency == 0:
                print(f"Roadmap graph has {len(roadmap_graph.graph)} nodes, "
                    f"{sum(len(neis) for neis in roadmap_graph.graph.values()) // 2} edges.")
                print(f"Orbital graph has {len(orbital_graph.orbit_graph)} orbits.")
                comp_time, path, distance = roadmap_graph.retrace_path(orbital_graph, q_s, env)
                time_stats["dijkstra"] += comp_time
                has_solution = record_solution(iter, path, distance)
                recorded_cost = float(distance) if has_solution else None
                cost_history.append((iter, recorded_cost))
                convergence_history.append({
                    "iteration": iter,
                    "elapsed_time": time.perf_counter() - planner_start_time,
                    "cost": recorded_cost,
                })
                print(
                    f"Convergence: iteration={iter}, "
                    f"elapsed={convergence_history[-1]['elapsed_time']:.4f}s, "
                    f"best_cost={recorded_cost if recorded_cost is not None else 'not found'}"
                )
                if has_solution:
                    print(f"Path from initial configuration to Goal has {len(path)} nodes. Cost: {distance:.4f}")
                else:
                    print("No path found from initial configuration to Goal.")
                    # print("Check orbit list:", [orb.object_states for orb in orbital_graph.graph])
                if path:
                    path_filename = (
                        f"{env_name}_planned_path_seed_{random_seed}_{alg_name}.txt"
                        if random_seed is not None
                        else f"{env_name}_planned_path_{alg_name}.txt"
                    )
                    save_path_to_file(path, path_cost=distance, filename=path_filename, random_seed=random_seed)

                print("--------------------------------------------------")
            completed_iterations = iter + 1
            if (
                planner_deadline is not None
                and time.perf_counter() >= planner_deadline
            ):
                stop_reason = "time_limit"
                print(
                    "Planning time limit reached after completing "
                    f"iteration {iter}."
                )
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("KeyboardInterrupt detected. Stopping the algorithm.")
            
    finally:
        ### Retrace path from G to q_s using the roadmap graph and orbital graph
        comp_time, path, distance = roadmap_graph.retrace_path(orbital_graph, q_s, env)
        time_stats["dijkstra"] += comp_time
        record_solution(completed_iterations, path, distance)
        print("Final path from initial configuration to Goal:")
        if path:        print(f"Path from initial configuration to Goal has {len(path)} nodes. Cost: {distance:.4f}")
        else:           print("No path found from initial configuration to Goal.")

        if show_final_path and path:
            path_filename = (
                f"{env_name}_planned_path_seed_{random_seed}_{alg_name}.txt"
                if random_seed is not None
                else f"{env_name}_planned_path_{alg_name}.txt"
            )
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

        if hasattr(roadmap_graph, "lazy_statistics"):
            time_stats.update(roadmap_graph.lazy_statistics())

        final_solution_cost = (
            float(distance)
            if path
            and distance is not None
            and np.isfinite(float(distance))
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
        return path, distance, time_stats, solution_statistics
    return path, distance, time_stats



if __name__ == "__main__":
    debugger_attached = sys.gettrace() is not None

    # Add/remove environments here. They will run sequentially in this order.
    environment_factories = [
        # ("simple_pick_place", lambda: SimplePickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("small_box_pick_place", lambda: SmallBoxPickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("shelf_pick_place", lambda: ShelfPickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=True)),
        # ("handover", lambda: HandoverEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("stacking_4_objects", lambda: StackingObjectsEnv(num_robots=2, num_mobjects=4, show_gui=False)),
        ("simple_motion_planning_2objs", lambda: SimpleMotionPlanningWithSubgoalEnv(num_robots=2, num_obstacles=4, num_mobjects=2)),
        # ("2r_pick_place_2_objects", lambda: SimplePickPlaceEnv(num_robots=2, num_mobjects=2, show_gui=False)),
        # ("3r_pick_place_6_objs", lambda: ThreeRobotsPickPlaceEnv(num_robots=3, num_mobjects=6, show_gui=False)),
        # ("4r_pick_place_6objs", lambda: FourRobotPickPlaceEnv(num_robots=4, num_mobjects=6, show_gui=False)),
    ]

    # Keep this harness aligned with drrt_star_main.py for ablation runs.
    trial = 5
    base_random_seed = 2
    lazy_collision_checking = False
    num_interior_samples = 20
    num_transition_samples = 20
    max_planner_iterations = 500   # max iterations
    max_planner_time_seconds = 600  # max time in seconds

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
        last_random_seed = base_random_seed + trial - 1
        statistics_filename = (
            f"{env_name}_trial_statistics_seeds_"
            f"{base_random_seed}-{last_random_seed}_{planner_variant}.txt"
        )
        statistics_metadata = {
            "Environment": env_name,
            "Environment class": type(env).__name__,
            "Algorithm": alg_name,
            "Planner variant": planner_variant,
            "Requested trials": trial,
            "Base random seed": base_random_seed,
            "Lazy collision checking": lazy_collision_checking,
            "Interior samples per iteration": num_interior_samples,
            "Transition samples per iteration": num_transition_samples,
            "Maximum planner iterations": max_planner_iterations,
            "Maximum planning time (seconds)": max_planner_time_seconds,
        }

        try:
            for i in range(trial):
                print(f"Trial {i+1}/{trial}")
                print(f"Using random seed: {random_seed}")
                _set_global_seed(random_seed)
                env.reset()
                start = time.time()
                (
                    path,
                    distance,
                    time_stats,
                    solution_statistics,
                ) = algo(
                    N_m=num_interior_samples,
                    N_t=num_transition_samples,
                    orbit_exploration_max_iter=max_planner_iterations,
                    max_iter=max_planner_iterations,
                    max_planning_time=max_planner_time_seconds,
                    print_frequency=10,
                    display_real_time=False,
                    show_final_path=False,
                    env=env,
                    env_name=env_name,
                    random_seed=random_seed,
                    show_cost_plot=True,
                    lazy_collision_checking=lazy_collision_checking,
                    return_solution_statistics=True,
                )
                end = time.time()
                trial_runtime = end - start
                best_cost = distance if distance is not None else float('inf')
                print(f"Trial {i+1} running time: {trial_runtime:.2f} seconds.")
                runtime.append(trial_runtime)
                distances.append(best_cost)
                trial_results.append({
                    "seed": random_seed,
                    "planner_variant": planner_variant,
                    "success": bool(
                        path
                        and distance is not None
                        and np.isfinite(float(distance))
                    ),
                    "runtime": trial_runtime,
                    "best_cost": best_cost,
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
