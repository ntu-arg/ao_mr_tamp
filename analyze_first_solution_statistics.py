#!/usr/bin/env python3
"""Summarize first-solution timing and cost from planner trial statistics.

Example:
    python3 analyze_first_solution_statistics.py \
        4robots_pick_place_4objs_trial_statistics_seeds_48-52_drrt_star.txt
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from statistics import mean, stdev


TRIAL_PATTERN = re.compile(
    r"(?ms)^\[TRIAL\]\s+\d+\s*$.*?(?=^\[TRIAL\]\s+\d+\s*$|\Z)"
)
NUMBER = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

# Two-sided 95% Student's t critical values for df=1..30. For larger samples,
# 1.96 is a close approximation to the limiting normal critical value.
T_95_CRITICAL = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def _read_number(block: str, *labels: str) -> float | None:
    """Return the first numeric value following one of ``labels``."""
    for label in labels:
        match = re.search(
            rf"(?mi)^\s*{re.escape(label)}\s*:\s*{NUMBER}", block
        )
        if match:
            value = float(match.group(1))
            return value if math.isfinite(value) else None
    return None


def _read_trial(block: str) -> dict:
    seed_match = re.search(r"(?mi)^\s*Random seed:\s*(\d+)", block)
    success_match = re.search(r"(?mi)^\s*Success:\s*(True|False)", block)

    return {
        "seed": int(seed_match.group(1)) if seed_match else None,
        "success": (
            success_match.group(1).lower() == "true"
            if success_match
            else False
        ),
        "first_time": _read_number(
            block,
            "Time to first solution",
            "First solution time",
        ),
        "first_cost": _read_number(
            block,
            "First solution cost",
        ),
        "final_cost": _read_number(
            block,
            "Final solution cost",
        ),
    }


def parse_statistics(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    trials = [_read_trial(block) for block in TRIAL_PATTERN.findall(text)]
    trials = [trial for trial in trials if trial["seed"] is not None]
    if not trials:
        raise ValueError(f"No trial records found in {path}")
    return trials


def format_value(value: float | None) -> str:
    return "not found" if value is None else f"{value:.6f}"


def confidence_interval_95(values: list[float]) -> tuple[float, float] | None:
    """Return a two-sided 95% Student's t confidence interval for the mean."""
    if len(values) < 2:
        return None
    degrees_of_freedom = len(values) - 1
    critical_value = (
        T_95_CRITICAL[degrees_of_freedom - 1]
        if degrees_of_freedom <= len(T_95_CRITICAL)
        else 1.96
    )
    margin = critical_value * stdev(values) / math.sqrt(len(values))
    average = mean(values)
    return average - margin, average + margin


def format_confidence_interval(values: list[float]) -> str:
    interval = confidence_interval_95(values)
    if interval is None:
        return "not estimable (need at least 2 runs)"
    return f"[{interval[0]:.6f}, {interval[1]:.6f}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report first-solution time and cost from trial statistics."
    )
    parser.add_argument("input_file", type=Path, help="Trial statistics file")
    args = parser.parse_args()

    if not args.input_file.is_file():
        parser.error(f"Input file does not exist: {args.input_file}")

    trials = parse_statistics(args.input_file)
    successful_trials = [
        trial
        for trial in trials
        if trial["success"]
        and trial["first_time"] is not None
        and trial["first_cost"] is not None
    ]
    final_cost_trials = [
        trial
        for trial in trials
        if trial["success"] and trial["final_cost"] is not None
    ]

    print(f"Input file: {args.input_file}")
    print()
    print("Per-seed first solution results:")
    print("Seed   Success   T_first (s)   J_first       J_final")
    print("-----  -------   ------------   ------------  ------------")
    for trial in trials:
        print(
            f"{trial['seed']:>4}   "
            f"{str(trial['success']):>7}   "
            f"{format_value(trial['first_time']):>12}   "
            f"{format_value(trial['first_cost']):>12}  "
            f"{format_value(trial['final_cost']):>12}"
        )

    times = [trial["first_time"] for trial in successful_trials]
    costs = [trial["first_cost"] for trial in successful_trials]
    final_costs = [trial["final_cost"] for trial in final_cost_trials]

    print()
    print(f"Successful runs used: {len(successful_trials)}/{len(trials)}")
    if not successful_trials:
        print("No successful runs with valid first-solution measurements.")
        return

    time_std = stdev(times) if len(times) > 1 else None
    cost_std = stdev(costs) if len(costs) > 1 else None
    print(f"T_first average over successful runs: {mean(times):.6f} s")
    print(
        "T_first standard deviation:            "
        f"{format_value(time_std)} s"
    )
    print(
        "T_first 95% confidence interval:        "
        f"{format_confidence_interval(times)} s"
    )
    print(f"J_first average over successful runs: {mean(costs):.6f}")
    print(f"J_first standard deviation:            {format_value(cost_std)}")
    print(f"J_first 95% confidence interval:        {format_confidence_interval(costs)}")
    print(
        f"J_final average over successful runs ({len(final_costs)} runs): "
        f"{mean(final_costs):.6f}"
        if final_costs
        else "J_final average over successful runs: not available"
    )
    final_cost_std = stdev(final_costs) if len(final_costs) > 1 else None
    print(f"J_final standard deviation:             {format_value(final_cost_std)}")
    print(
        "J_final 95% confidence interval:        "
        f"{format_confidence_interval(final_costs)}"
    )


if __name__ == "__main__":
    main()
