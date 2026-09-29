import matplotlib.pyplot as plt
import numpy as np


def disconnect_all_pybullet_clients(max_client_id=64):
    """Disconnect lingering PyBullet clients during planner cleanup."""
    import pybullet as p

    for client_id in range(max_client_id):
        try:
            if p.isConnected(client_id):
                p.disconnect(client_id)
        except Exception:
            continue


def cleanup_between_environments(max_client_id=64):
    """Close plotting state, disconnect PyBullet, and collect garbage."""
    import gc

    plt.close("all")
    disconnect_all_pybullet_clients(max_client_id)
    gc.collect()


def set_global_seed(seed):
    """Seed the Python and NumPy random generators."""
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)


def save_trial_statistics(trial_results, filename, metadata=None):
    """Write reproducible per-trial statistics and aggregate averages.

    Each result provides ``seed``, ``runtime``, ``best_cost``, and
    ``time_stats``. Optional fields include ``success``, ``path_length``,
    ``solution_statistics``, and ``error``. Rewriting the complete report
    after every completed trial keeps partial multi-seed experiments usable if
    a later run is interrupted.
    """
    separator = "=" * 72
    metadata = metadata or {}

    with open(filename, "w", encoding="utf-8") as output:
        output.write(f"{separator}\n")
        output.write("DRRT* MULTI-TRIAL STATISTICS\n")
        output.write(f"{separator}\n")
        for name, value in metadata.items():
            output.write(f"{name}: {value}\n")
        output.write(f"Completed trials: {len(trial_results)}\n\n")

        for trial_number, result in enumerate(trial_results, start=1):
            output.write(f"{separator}\n")
            output.write(f"[TRIAL] {trial_number}\n")
            output.write(f"Random seed: {result['seed']}\n")
            if result.get("planner_variant") is not None:
                output.write(
                    f"Planner variant: {result['planner_variant']}\n"
                )
            best_cost = result.get("best_cost")
            finite_cost = (
                best_cost is not None and np.isfinite(float(best_cost))
            )
            success = bool(result.get("success", finite_cost))
            output.write(f"Success: {success}\n")
            output.write(
                f"Path configurations: {int(result.get('path_length', 0))}\n"
            )
            output.write(
                f"Total trial runtime: {float(result['runtime']):.4f} seconds\n"
            )
            output.write(
                "Best path cost: "
                f"{float(best_cost) if finite_cost else float('inf'):.4f}\n"
            )
            if result.get("error"):
                output.write(f"Error: {result['error']}\n")
            output.write("Timing statistics (in seconds):\n")
            for name, value in result["time_stats"].items():
                output.write(f"  {name}: {float(value):.4f}\n")

            if "solution_statistics" in result:
                solution_statistics = result["solution_statistics"]
                first_solution_iteration = solution_statistics.get(
                    "first_solution_iteration"
                )
                first_solution_time = solution_statistics.get(
                    "first_solution_time"
                )
                first_solution_cost = solution_statistics.get(
                    "first_solution_cost"
                )
                final_solution_cost = solution_statistics.get(
                    "final_solution_cost", best_cost if finite_cost else None
                )
                output.write("Solution statistics:\n")
                output.write(
                    "  Stop reason: "
                    f"{solution_statistics.get('stop_reason', 'unknown')}\n"
                )
                output.write(
                    "  Completed planner iterations: "
                    f"{int(solution_statistics.get('completed_iterations', 0))}\n"
                )
                configured_time_limit = solution_statistics.get(
                    "max_planning_time"
                )
                output.write(
                    "  Planning time limit: "
                    + (
                        f"{float(configured_time_limit):.4f} seconds\n"
                        if configured_time_limit is not None
                        else "disabled\n"
                    )
                )
                output.write(
                    "  First solution iteration: "
                    + (
                        f"{int(first_solution_iteration)}\n"
                        if first_solution_iteration is not None
                        else "not found\n"
                    )
                )
                output.write(
                    "  Time to first solution: "
                    + (
                        f"{float(first_solution_time):.4f} seconds\n"
                        if first_solution_time is not None
                        else "not found\n"
                    )
                )
                output.write(
                    "  First solution cost: "
                    + (
                        f"{float(first_solution_cost):.4f}\n"
                        if first_solution_cost is not None
                        else "not found\n"
                    )
                )
                output.write(
                    "  Final solution cost: "
                    + (
                        f"{float(final_solution_cost):.4f}\n"
                        if final_solution_cost is not None
                        else "not found\n"
                    )
                )
                output.write(
                    "  Convergence over time "
                    "(iteration, elapsed seconds, best path cost):\n"
                )
                convergence = solution_statistics.get("convergence", [])
                if convergence:
                    for point in convergence:
                        point_cost = point.get("cost")
                        formatted_cost = (
                            f"{float(point_cost):.4f}"
                            if point_cost is not None
                            else "not found"
                        )
                        output.write(
                            f"    {int(point['iteration'])}, "
                            f"{float(point['elapsed_time']):.4f}, "
                            f"{formatted_cost}\n"
                        )
                else:
                    output.write("    no solution samples\n")
            output.write("\n")

        if trial_results:
            runtimes = np.asarray(
                [result["runtime"] for result in trial_results], dtype=float
            )
            successful_results = [
                result
                for result in trial_results
                if result.get("success", False)
                and result.get("best_cost") is not None
                and np.isfinite(float(result["best_cost"]))
            ]
            successful_costs = np.asarray(
                [result["best_cost"] for result in successful_results],
                dtype=float,
            )

            output.write(f"{separator}\n")
            output.write("[SUMMARY]\n")
            output.write(
                "Random seeds: "
                + ", ".join(str(result["seed"]) for result in trial_results)
                + "\n"
            )
            output.write(
                f"Successful trials: {len(successful_results)}/"
                f"{len(trial_results)}\n"
            )
            output.write(
                f"Average total runtime: {np.mean(runtimes):.4f} seconds\n"
            )
            output.write(
                f"Total runtime standard deviation: {np.std(runtimes):.4f} seconds\n"
            )

            timing_names = list(dict.fromkeys(
                name
                for result in trial_results
                for name in result["time_stats"]
            ))
            output.write("Average timing statistics (in seconds):\n")
            for name in timing_names:
                values = np.asarray(
                    [
                        result["time_stats"][name]
                        for result in trial_results
                        if name in result["time_stats"]
                    ],
                    dtype=float,
                )
                output.write(
                    f"  {name}: mean={np.mean(values):.4f}, "
                    f"std={np.std(values):.4f}\n"
                )

            if successful_costs.size:
                output.write(
                    f"Average best path cost over {successful_costs.size} "
                    f"successful trials: {np.mean(successful_costs):.4f}\n"
                )
                output.write(
                    "Best path cost standard deviation: "
                    f"{np.std(successful_costs):.4f}\n"
                )
            else:
                output.write("Average best path cost: no successful trials\n")

    print(f"Trial statistics saved to {filename}")



def get_orbit_path_from_path(path):
    
    orbit_path = []         # list of Orbit:: corresponding to the configurations in the path
    current_orbit = _get_parent_orbits(path[0])
    if isinstance(current_orbit, list):
        # assumming the start configuration is not transition configuration, it should have only one parent orbit
        current_orbit = current_orbit[0] if current_orbit else None
    orbit_path.append(current_orbit)

    for i, config in enumerate(path):
        if i==0:
            continue 

        parent_orbits = _get_parent_orbits(config)
        if current_orbit not in parent_orbits:
            # if current orbit is not in the list of parent orbits, last one should be the transition, current one can be transition of interior
            next_orbit = None
            for parent in parent_orbits:
                if parent != current_orbit and parent in _get_parent_orbits(path[i-1]):
                    next_orbit = parent
                    break
            if next_orbit is not None:
                current_orbit = next_orbit
                orbit_path.append(current_orbit)  # Append the current orbit after potentially updating it
    return orbit_path


def update_roadmap_graph_with_orbit_path(roadmap_graph, orbit_path):
    """
    Update the roadmap graph to keep only the configurations that are part of the given orbit path.

    :param roadmap_graph: The roadmap graph to be updated.
    :param orbit_path: List of Orbit objects representing the path.
    """

    for node in list(roadmap_graph.graph.keys()):
        if all(parent_orbit not in orbit_path for parent_orbit in _get_parent_orbits(node)):
            roadmap_graph.remove_node(node)

    


def _get_parent_orbits(config):
    if config.parent_orbit is None:
        return []
    if isinstance(config.parent_orbit, list):
        return config.parent_orbit
    return [config.parent_orbit]


def _write_orbit_info(f, orbit):
    if orbit is None:
        f.write("Orbit Information: None\n")
        return

    orbit_id = id(orbit)
    robot_states_str = {r_id: s.name for r_id, s in orbit.robot_states.items()}
    object_states_str = {o_id: s.name for o_id, s in orbit.object_states.items()}
    attachments_str = {
        r_id: (f"object {obj_id}" if obj_id is not None else "None")
        for r_id, obj_id in orbit.attachments.items()
    }

    f.write(f"Orbit Information: id={orbit_id}, hex={hex(orbit_id)}\n")
    f.write(f"  Robot States: {robot_states_str}\n")
    f.write(f"  Object States: {object_states_str}\n")
    f.write(f"  Attachments: {attachments_str}\n")
    f.write(f"  Grasp Pose Objects: {sorted(list(orbit.grasp_poses.keys()))}\n")
    f.write(f"  Placement Pose Objects: {sorted(list(orbit.placement_poses.keys()))}\n")
    if orbit.grasp_poses:
        f.write("  Grasp Pose Matrices:\n")
        for obj_id, pose in orbit.grasp_poses.items():
            f.write(f"    Object {obj_id}:\n")
            f.write(f"          {np.array_str(np.array(pose), precision=6)}\n")
    if orbit.placement_poses:
        f.write("  Placement Pose Matrices:\n")
        for obj_id, pose in orbit.placement_poses.items():
            f.write(f"    Object {obj_id}:\n")
            f.write(f"          {np.array_str(np.array(pose), precision=6)}\n")


def _write_configuration_details(f, q_label, global_idx, config):
    f.write(f"  {q_label} (Configuration {global_idx}):\n")

    joint_angles = np.round(np.array(config.joint_angles, dtype=float), 4)
    f.write(f"    Joint Angles: {joint_angles.tolist()}\n")
    # f.write(f"    Shape: {joint_angles.shape}\n")
    # if joint_angles.ndim == 2:
    #     for r_id, robot_angles in enumerate(joint_angles):
    #         f.write(f"    Robot {r_id}:\n")
    #         for j_id, angle in enumerate(robot_angles):
    #             f.write(f"      Joint {j_id}: {angle:.4f} rad ({np.degrees(angle):.2f} deg)\n")

    f.write(f"    Attachments: {config.attachments}\n")
    f.write(f"    Is Transition: {config.is_transition}\n")

    parent_orbits = _get_parent_orbits(config)
    parent_ids = [id(orb) for orb in parent_orbits]
    f.write(f"    Parent Orbit IDs: {parent_ids}\n")

    if config.grasp_poses:
        f.write(f"    Grasp Poses: {sorted(list(config.grasp_poses.keys()))}\n")
        for obj_id, pose in config.grasp_poses.items():
            f.write(f"      Object {obj_id}:\n")
            f.write(f"          {np.array_str(np.array(pose), precision=6)}\n")
    if config.placement_poses:
        f.write(f"    Placement Poses: {sorted(list(config.placement_poses.keys()))}\n")
        for obj_id, pose in config.placement_poses.items():
            f.write(f"      Object {obj_id}:\n")
            f.write(f"          {np.array_str(np.array(pose), precision=6)}\n")
    if config.pre_pose:
        f.write(f"    Pre Poses:\n")
        for r_id, pose in enumerate(config.pre_pose):
            f.write(f"      Robot {r_id}: {np.array_str(np.array(pose), precision=6)}\n")


def _build_orbit_segments(path):
    if not path:
        return []

    first_parents = _get_parent_orbits(path[0]) # start configuration should have only one parent orbit
    current_orbit = first_parents[0] if first_parents else None
    segments = [{"orbit": current_orbit, "indices": [0], "bridge_index": None}]

    for i in range(1, len(path)):
        pre_node_parents = _get_parent_orbits(path[i - 1])
        parents = _get_parent_orbits(path[i])
        in_current_orbit = (current_orbit in parents) or (current_orbit is None and len(parents) == 0)

        if in_current_orbit:
            if segments[-1]["indices"][-1] != i:
                segments[-1]["indices"].append(i)
            continue

        next_orbit = None
        for parent in parents:
            if parent != current_orbit and parent in pre_node_parents:
                next_orbit = parent
                break 
        prev_idx = i - 1
        segments.append({"orbit": next_orbit, "indices": [i], "bridge_index": prev_idx})
        current_orbit = next_orbit

    return segments


def save_path_to_file(path, path_cost, filename="path_output.txt", random_seed=None):
    """
    Save a planned path to a text file for inspection.
    
    :param path: list of Configuration objects
    :param filename: output filename
    """
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"PLANNED PATH WITH {len(path)} CONFIGURATIONS. COST: {path_cost:.4f}\n")
        if random_seed is not None:
            f.write(f"RANDOM SEED: {random_seed}\n")
        f.write("=" * 80 + "\n\n")

        orbit_segments = _build_orbit_segments(path)

        for seg_idx, segment in enumerate(orbit_segments, start=1):
            orbit = segment["orbit"]
            indices = segment["indices"]
            bridge_index = segment.get("bridge_index")
            q_chain = "->".join([f"q{idx + 1}" for idx in indices])

            f.write(f"Orbit{seg_idx}: {{{q_chain}}}\n")
            _write_orbit_info(f, orbit)
            if bridge_index is not None:
                f.write(f"  Entry Bridge: q{bridge_index + 1} (shared boundary from previous orbit)\n")
            f.write("  Configurations in this orbit path:\n")

            for idx in indices:
                _write_configuration_details(f, q_label=f"q{idx + 1}", global_idx=idx, config=path[idx])

            f.write("\n")
    
    print(f"Path saved to {filename}")


def save_orbit_graph_to_file(orbit_graph, filename="orbit_graph.txt"):
    """
    Save all orbit information and orbit-graph connections to a text file.

    :param orbit_graph: OrbitGraph object
    :param filename: output filename
    """
    orbit_nodes = list(orbit_graph.orbit_graph.keys())
    orbit_index = {orbit: idx for idx, orbit in enumerate(orbit_nodes)}

    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ORBIT GRAPH WITH {len(orbit_nodes)} ORBITS\n")
        modal_graph = getattr(orbit_graph, "modal_graph", None)
        if modal_graph is not None:
            f.write(f"CANONICAL MODES: {len(modal_graph.modes_by_key)}\n")
            f.write(f"MODAL STATES: {len(modal_graph)}\n")
        edge_count = sum(len(neis) for neis in orbit_graph.orbit_graph.values()) // 2
        f.write(f"UNDIRECTED EDGES: {edge_count}\n")
        f.write("=" * 80 + "\n\n")

        f.write("ORBITS\n")
        f.write("-" * 80 + "\n")
        for orbit in orbit_nodes:
            idx = orbit_index[orbit]
            orbit_id = id(orbit)
            robot_states_str = {r_id: s.name for r_id, s in orbit.robot_states.items()}
            object_states_str = {o_id: s.name for o_id, s in orbit.object_states.items()}
            attachments_str = {
                r_id: (f"object {obj_id}" if obj_id is not None else "None")
                for r_id, obj_id in orbit.attachments.items()
            }

            f.write(f"Orbit {idx} (id={orbit_id}, hex={hex(orbit_id)})\n")
            f.write(f"  Robot States: {robot_states_str}\n")
            f.write(f"  Object States: {object_states_str}\n")
            f.write(f"  Attachments: {attachments_str}\n")
            parent_mode = getattr(orbit, "parent_mode", None)
            parent_modal_state = getattr(orbit, "parent_modal_state", None)
            if parent_mode is not None:
                f.write(f"  Parent Mode: {parent_mode.attachments}\n")
            if parent_modal_state is not None:
                f.write(
                    "  Modal Task Progress: "
                    f"{parent_modal_state.at_goal}\n"
                )
            f.write(f"  Is Goal Orbit: {orbit.is_goal_orbit}\n")
            f.write(f"  Number of Configurations: {len(orbit.configurations)}\n")
            f.write(f"  Grasp Pose Objects: {sorted(list(orbit.grasp_poses.keys()))}\n")
            f.write(f"  Placement Pose Objects: {sorted(list(orbit.placement_poses.keys()))}\n")
            # if orbit.placement_poses:
            #     f.write("  Placement Poses (translation only):\n")
            #     for obj_id, pose in orbit.placement_poses.items():
            #         # TODO: in 2d environment, the placement pose is just simply [x,y]

            #         xyz = np.round(np.array(pose[:3, 3], dtype=float), 4).tolist()
            #         f.write(f"    Object {obj_id}: {xyz}\n")
            # f.write("\n")

            if orbit.placement_poses:
                f.write("  Placement Poses (translation only):\n")
                for obj_id, pose in orbit.placement_poses.items():
                    # Convert to numpy array to handle formatting uniformly
                    pose_np = np.array(pose, dtype=float)
                    
                    if pose_np.ndim == 1:
                        # Case 1: simple 2D environment pose, e.g., [x, y]
                        xy_or_xyz = np.round(pose_np, 4).tolist()
                    else:
                        # Case 2: full 3D transformation matrix (4x4) -> write (x,y,z) position only
                        xy_or_xyz = np.round(pose_np[:3, 3], 4).tolist()
                        
                    f.write(f"    Object {obj_id}: {xy_or_xyz}\n")
            f.write("\n")

        f.write("CONNECTIONS (Adjacency)\n")
        f.write("-" * 80 + "\n")
        for orbit in orbit_nodes:
            idx = orbit_index[orbit]
            neighbors = orbit_graph.orbit_graph.get(orbit, [])
            neighbor_indices = sorted(orbit_index[n] for n in neighbors)
            f.write(f"Orbit {idx} -> {neighbor_indices}\n")

        f.write("\nUNDIRECTED EDGE LIST\n")
        f.write("-" * 80 + "\n")
        seen = set()
        for orbit in orbit_nodes:
            u = orbit_index[orbit]
            for neig in orbit_graph.orbit_graph.get(orbit, []):
                v = orbit_index[neig]
                edge = (min(u, v), max(u, v))
                if edge in seen:
                    continue
                seen.add(edge)
                f.write(f"{edge[0]} <-> {edge[1]}\n")

        if modal_graph is not None:
            modal_states = list(modal_graph.modal_graph)
            modal_state_index = {
                state: index
                for index, state in enumerate(modal_states)
            }
            f.write("\nMODAL STATES\n")
            f.write("-" * 80 + "\n")
            for state in modal_states:
                state_index = modal_state_index[state]
                associated_orbits = sorted(
                    orbit_index[orbit]
                    for orbit in modal_graph.orbits_by_state[state]
                )
                neighbor_states = sorted(
                    modal_state_index[neighbor]
                    for neighbor in modal_graph.get_neighbors(state)
                )
                f.write(
                    f"Modal State {state_index}: "
                    f"mode={state.mode.attachments}, "
                    f"at_goal={state.at_goal}, "
                    f"orbits={associated_orbits}, "
                    f"neighbors={neighbor_states}\n"
                )

    print(f"Orbit graph saved to {filename}")


def plot_path_cost(costs, print_frequency=1, filename="convergence.png"):
    """
    Plot path cost over iterations.

    :param costs: list of (iteration, cost) tuples. cost can be None if no path was found.
    :param print_frequency: the iteration step between recorded costs.
    :param filename: the filename to save the plot.
    """
    iterations = [it for it, c in costs if c is not None]
    values = [c for _, c in costs if c is not None]

    if not values:
        print("No path costs to plot.")
        return

    plt.figure()
    plt.plot(iterations, values, '-o', ms=3, linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Path Cost')
    plt.title('Path Cost over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
