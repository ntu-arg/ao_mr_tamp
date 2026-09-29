#!/usr/bin/env python3
"""Plot multi-trial PRM convergence and overlay saved robot paths.

Example:
    python3 plot_prm_trial_results.py \\
        simple_motion_planning_2objs_trial_statistics_seeds_2-6_PRM_star.txt

Two images are written beside the statistics file:
    *_convergence_mean_95ci.png
    *_all_trial_paths.png
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Circle, Rectangle
import numpy as np

from analyze_first_solution_statistics import confidence_interval_95


TRIAL_PATTERN = re.compile(
    r"(?ms)^\[TRIAL\]\s+\d+\s*$.*?(?=^\[TRIAL\]\s+\d+\s*$|\Z)"
)
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CONVERGENCE_LINE = re.compile(
    rf"^\s*(\d+)\s*,\s*({NUMBER})\s*,\s*(not found|{NUMBER})\s*$",
    re.IGNORECASE,
)

# Edit this to change the convergence plot's iteration limit. It can also be
# overridden for one run with --max-iterations.
MAX_ITERATIONS = 300
ZOOM_START_ITERATION = 40

# Rectangles for the environment layout used to generate the saved PRM paths.
# The trial-statistics/path files do not serialize obstacle geometry, so update
# this list if plotting results from a different environment layout.
# Each entry is (lower-left x, lower-left y, width, height).
OBSTACLES = (
    (4.0, 1.0, 2.0, 1.0),
    (9.0, 1.2, 2.0, 1.0),
    (6.0, 3.0, 2.0, 1.0),
    (11.0, 4.2, 2.0, 1.0),
)


def read_label_number(text: str, label: str) -> float | None:
    match = re.search(
        rf"(?mi)^\s*{re.escape(label)}\s*:\s*({NUMBER})", text
    )
    return float(match.group(1)) if match else None


def parse_statistics(path: Path) -> tuple[str, str, list[dict]]:
    text = path.read_text(encoding="utf-8")
    environment_match = re.search(r"(?mi)^Environment:\s*(\S+)", text)
    variant_match = re.search(r"(?mi)^Planner variant:\s*(\S+)", text)
    environment = environment_match.group(1) if environment_match else path.name.split("_trial_statistics")[0]
    variant = variant_match.group(1) if variant_match else "PRM_star"

    trials = []
    for block in TRIAL_PATTERN.findall(text):
        seed_match = re.search(r"(?mi)^\s*Random seed:\s*(\d+)", block)
        success_match = re.search(r"(?mi)^\s*Success:\s*(True|False)", block)
        if seed_match is None:
            continue

        convergence = []
        in_convergence = False
        for line in block.splitlines():
            if re.match(r"^\s*Convergence over time", line, re.IGNORECASE):
                in_convergence = True
                continue
            if in_convergence:
                match = CONVERGENCE_LINE.match(line)
                if match:
                    value = (
                        float(match.group(3))
                        if match.group(3).lower() != "not found"
                        else math.nan
                    )
                    convergence.append((
                        int(match.group(1)),
                        float(match.group(2)),
                        value,
                    ))
                elif convergence:
                    break

        trials.append({
            "seed": int(seed_match.group(1)),
            "success": bool(
                success_match and success_match.group(1).lower() == "true"
            ),
            "final_cost": read_label_number(block, "Final solution cost"),
            "convergence": convergence,
        })

    if not trials:
        raise ValueError(f"No trial records found in {path}")
    return environment, variant, trials


def parse_numpy_array(text: str) -> list[float]:
    values = np.fromstring(text.replace(",", " "), sep=" ")
    if values.size == 0:
        raise ValueError(f"Could not parse coordinate array: [{text}]")
    return values.tolist()


def placement_poses_from_orbit_block(block: str) -> dict[int, list[float]]:
    heading = re.search(r"(?ms)^\s*Placement Pose Matrices:\s*\n(.*?)(?=^\s*(?:Configurations in this orbit path|Entry Bridge|Grasp Pose Matrices|$))", block)
    if heading is None:
        return {}
    section = heading.group(1)
    poses = {}
    for match in re.finditer(
        rf"(?ms)^\s*Object\s+(\d+):\s*\n\s*\[([^\]]+)\]",
        section,
    ):
        poses[int(match.group(1))] = parse_numpy_array(match.group(2))
    return poses


def parse_saved_path(path: Path) -> tuple[list[np.ndarray], dict, dict]:
    text = path.read_text(encoding="utf-8")
    orbit_blocks = re.findall(
        r"(?ms)^Orbit\d+:.*?(?=^Orbit\d+:|\Z)", text
    )
    if not orbit_blocks:
        raise ValueError(f"No orbit/configuration sections in {path}")

    configurations = []
    for match in re.finditer(r"(?m)^\s*Joint Angles:\s*(.+)$", text):
        try:
            joint_angles = np.asarray(ast.literal_eval(match.group(1)), dtype=float)
        except (ValueError, SyntaxError) as error:
            raise ValueError(
                f"Could not parse Joint Angles in {path}: {match.group(1)}"
            ) from error
        if joint_angles.ndim != 2 or joint_angles.shape[1] < 2:
            raise ValueError(f"Expected 2-D robot poses in {path}")
        configurations.append(joint_angles[:, :2])

    start_poses = placement_poses_from_orbit_block(orbit_blocks[0])
    goal_poses = placement_poses_from_orbit_block(orbit_blocks[-1])
    if not configurations:
        raise ValueError(f"No joint configurations in {path}")
    if not start_poses or not goal_poses:
        raise ValueError(
            f"Could not recover object start/goal positions from {path}"
        )
    return configurations, start_poses, goal_poses


def plot_convergence(
    trials: list[dict], output_path: Path, max_iterations: int = MAX_ITERATIONS
) -> None:
    successful = [
        trial for trial in trials
        if trial["success"] and trial["final_cost"] is not None
        and trial["convergence"]
    ]
    if not successful:
        raise ValueError("No successful trials with convergence samples found")

    last_iteration = max(
        max(iteration for iteration, _, _ in trial["convergence"])
        for trial in successful
    )
    iteration_limit = min(max_iterations, last_iteration)
    iteration_grid = np.arange(iteration_limit + 1, dtype=float)
    curves = []

    # Treat each log as a right-continuous best-so-far step function. Before
    # its first solution, a trial is missing; after its last sample, retain its
    # final incumbent cost.
    for trial in successful:
        samples = sorted(trial["convergence"])
        sample_iterations = np.asarray([sample[0] for sample in samples])
        sample_costs = np.asarray([sample[2] for sample in samples])
        curve = np.full(iteration_grid.shape, np.nan, dtype=float)
        indices = np.searchsorted(
            sample_iterations, iteration_grid, side="right"
        ) - 1
        valid = indices >= 0
        curve[valid] = sample_costs[indices[valid]]
        after_last = iteration_grid > sample_iterations[-1]
        curve[after_last] = float(trial["final_cost"])
        curves.append(curve)

    values = np.asarray(curves)
    mean_cost = np.full(iteration_grid.shape, np.nan)
    ci_low = np.full(iteration_grid.shape, np.nan)
    ci_high = np.full(iteration_grid.shape, np.nan)
    for index in range(len(iteration_grid)):
        available = values[:, index]
        available = available[np.isfinite(available)]
        if not available.size:
            continue
        mean_cost[index] = float(np.mean(available))
        interval = confidence_interval_95(available.tolist())
        if interval is not None:
            ci_low[index], ci_high[index] = interval

    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    line_color = "#245B78"
    band_color = "#4C9BB8"
    ax.plot(
        iteration_grid, mean_cost, color=line_color, linewidth=2.4,
        label="Mean cost",
    )
    ax.fill_between(
        iteration_grid, ci_low, ci_high, color=band_color, alpha=0.24,
        linewidth=0, label="95% CI",
    )
    # Do not let Matplotlib add margins beyond the requested iteration cutoff.
    ax.set_xlim(0, max_iterations)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best path cost")
    ax.grid(True, alpha=0.25)
    ax.axhline(
        22.08, color="red", linestyle="--", linewidth=1.8,
        label="Optimal value (22.08)",
    )
    ax.legend(frameon=True, loc="upper left")

    # The first few incumbent costs vary widely between seeds, which makes
    # later refinement hard to see at the same scale. Keep the full CI visible
    # in the main axes and add a late-iteration zoom without clipping it.
    zoom_mask = (
        (iteration_grid >= min(ZOOM_START_ITERATION, iteration_limit))
        & np.isfinite(mean_cost)
        & np.isfinite(ci_low)
        & np.isfinite(ci_high)
    )
    if np.count_nonzero(zoom_mask) >= 2:
        zoom_ax = ax.inset_axes([0.48, 0.48, 0.48, 0.46])
        zoom_ax.fill_between(
            iteration_grid[zoom_mask],
            ci_low[zoom_mask],
            ci_high[zoom_mask],
            color=band_color,
            alpha=0.24,
            linewidth=0,
        )
        zoom_ax.plot(
            iteration_grid[zoom_mask],
            mean_cost[zoom_mask],
            color=line_color,
            linewidth=1.9,
        )
        zoom_ax.set_xlim(
            float(iteration_grid[zoom_mask][0]),
            float(iteration_grid[zoom_mask][-1]),
        )
        zoom_min = float(np.min(ci_low[zoom_mask]))
        zoom_max = float(np.max(ci_high[zoom_mask]))
        zoom_padding = max(0.15, 0.08 * (zoom_max - zoom_min))
        zoom_ax.set_ylim(zoom_min - zoom_padding, zoom_max + zoom_padding)
        zoom_ax.grid(True, alpha=0.2)
        zoom_ax.tick_params(labelsize=8)
        zoom_ax.set_title("Later refinement", fontsize=9, pad=3)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def interpolate_path(configurations: list[np.ndarray], points_per_edge: int = 12) -> np.ndarray:
    pieces = []
    for first, second in zip(configurations, configurations[1:]):
        if first.shape != second.shape:
            raise ValueError("Robot count changed within a saved path")
        alpha = np.linspace(0.0, 1.0, points_per_edge, endpoint=False)
        pieces.append(first[None, :, :] + alpha[:, None, None] * (second - first)[None, :, :])
    pieces.append(configurations[-1][None, :, :])
    return np.concatenate(pieces, axis=0)


def plot_all_paths(path_records: list[dict], output_path: Path) -> None:
    first_starts = path_records[0]["starts"]
    first_goals = path_records[0]["goals"]
    for record in path_records[1:]:
        if record["starts"].keys() != first_starts.keys() or record["goals"].keys() != first_goals.keys():
            raise ValueError("Saved paths disagree on object IDs")
        for object_id in first_starts:
            if not np.allclose(record["starts"][object_id], first_starts[object_id]):
                raise ValueError("Saved trials use different object start positions")
            if not np.allclose(record["goals"][object_id], first_goals[object_id]):
                raise ValueError("Saved trials use different object goal positions")

    robot_count = path_records[0]["configurations"][0].shape[0]
    for record in path_records:
        if record["configurations"][0].shape[0] != robot_count:
            raise ValueError("Saved paths have different robot counts")

    colors = ["#2676B8", "#26A65B", "#9B59B6", "#E67E22"]
    all_points = []
    fig, ax = plt.subplots(figsize=(10.0, 6.4), constrained_layout=True)

    for obstacle_x, obstacle_y, obstacle_width, obstacle_height in OBSTACLES:
        ax.add_patch(Rectangle(
            (obstacle_x, obstacle_y),
            obstacle_width,
            obstacle_height,
            facecolor="#777777",
            edgecolor="#333333",
            linewidth=1.0,
            alpha=0.48,
            zorder=1,
        ))
        all_points.extend((
            np.array([obstacle_x, obstacle_y], dtype=float)[None, :],
            np.array(
                [obstacle_x + obstacle_width, obstacle_y + obstacle_height],
                dtype=float,
            )[None, :],
        ))

    for record in path_records:
        trajectory = interpolate_path(record["configurations"])
        all_points.append(trajectory.reshape(-1, 2))
        for robot_id in range(robot_count):
            ax.plot(
                trajectory[:, robot_id, 0],
                trajectory[:, robot_id, 1],
                color=colors[robot_id % len(colors)],
                linewidth=1.55,
                alpha=0.42,
                solid_capstyle="round",
                zorder=2,
            )

    initial_trajectory = interpolate_path(path_records[0]["configurations"])
    for robot_id in range(robot_count):
        robot_position = initial_trajectory[0, robot_id]
        ax.add_patch(Circle(
            robot_position,
            radius=0.2,
            facecolor=colors[robot_id % len(colors)],
            edgecolor="black",
            linewidth=0.75,
            zorder=4,
        ))

    object_colors = ["#D62728", "#FF7F0E", "#BCBD22", "#8C564B"]
    for object_id, pose in first_starts.items():
        color = object_colors[object_id % len(object_colors)]
        ax.scatter(
            pose[0], pose[1], marker="^", s=170, color=color,
            edgecolor="black", linewidth=0.75, zorder=5,
        )
        all_points.append(np.asarray(pose, dtype=float)[None, :])
    for object_id, pose in first_goals.items():
        color = object_colors[object_id % len(object_colors)]
        ax.scatter(
            pose[0], pose[1], marker="*", s=280, color=color,
            edgecolor="black", linewidth=0.75, zorder=6,
        )
        all_points.append(np.asarray(pose, dtype=float)[None, :])

    points = np.concatenate(all_points, axis=0)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    # Keep only a small visual margin around the geometry so the saved figure
    # does not contain a large unused white border.
    x_pad = max(0.3, 0.03 * (x_max - x_min))
    y_pad = max(0.3, 0.04 * (y_max - y_min))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.2)

    robot_handles = tuple(
        Line2D([], [], color=colors[index % len(colors)], linewidth=2.4)
        for index in range(robot_count)
    )
    start_handle = Line2D(
        [], [], marker="^", linestyle="None", markerfacecolor="#888888",
        markeredgecolor="black", markersize=8,
    )
    goal_handle = Line2D(
        [], [], marker="*", linestyle="None", markerfacecolor="#888888",
        markeredgecolor="black", markersize=11,
    )
    robot_handle = Line2D(
        [], [], marker="o", linestyle="None", markerfacecolor="#888888",
        markeredgecolor="black", markersize=9,
    )
    ax.legend(
        [robot_handles, robot_handle, start_handle, goal_handle],
        ["Robot paths", "Robot (radius 0.2)", "Object starts", "Object goals"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
        loc="best",
        frameon=True,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PRM multi-trial convergence and overlaid saved paths."
    )
    parser.add_argument("statistics_file", type=Path)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Convergence plot cutoff (default: {MAX_ITERATIONS})",
    )
    parser.add_argument("--convergence-output", type=Path)
    parser.add_argument("--paths-output", type=Path)
    args = parser.parse_args()

    statistics_file = args.statistics_file.resolve()
    if not statistics_file.is_file():
        parser.error(f"Statistics file does not exist: {statistics_file}")

    environment, variant, trials = parse_statistics(statistics_file)
    path_records = []
    missing_paths = []
    for trial in trials:
        if not trial["success"]:
            continue
        path_file = statistics_file.parent / (
            f"{environment}_planned_path_seed_{trial['seed']}_{variant}.txt"
        )
        if not path_file.is_file():
            missing_paths.append(path_file.name)
            continue
        configurations, starts, goals = parse_saved_path(path_file)
        path_records.append({
            "seed": trial["seed"],
            "configurations": configurations,
            "starts": starts,
            "goals": goals,
        })

    convergence_output = args.convergence_output or statistics_file.with_name(
        f"{statistics_file.stem}_convergence_mean_95ci.png"
    )
    paths_output = args.paths_output or statistics_file.with_name(
        f"{statistics_file.stem}_all_trial_paths.png"
    )

    if args.max_iterations < 0:
        parser.error("--max-iterations must be non-negative")
    plot_convergence(trials, convergence_output, args.max_iterations)
    print(f"Saved convergence plot: {convergence_output}")
    if path_records:
        plot_all_paths(path_records, paths_output)
        print(f"Saved overlaid path plot ({len(path_records)} successful trials): {paths_output}")
    else:
        print("No saved successful planned paths found; skipped path plot.")
    if missing_paths:
        print("Missing saved paths:")
        for name in missing_paths:
            print(f"  {name}")


if __name__ == "__main__":
    main()
