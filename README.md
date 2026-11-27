# Asymptotically Optimal Multi-Robot Task and Motion Planning

[Blog](https://ntu-arg.github.io/ao_mr_tamp/)

## Highlights

- An asymptotically optimal framework for multi-robot task and motion planning.
- Implicit tensor-product search over constituent roadmaps enables scalable multi-robot planning.
- Modal- and orbit-guided exploration efficiently navigates long-horizon manipulation tasks.


## Provided planners

The central comparison is:

| Method | Search representation | Additional |
|---|---|---|
| **Guided dRRT\*** (`planners/drrt_star_main.py`) | Constituent roadmaps with implicit tensor-product search | Orbit, modal-state, frontier, and transition guidance |
| **Original dRRT\*** (`planners/drrt_star_original.py`) | Constituent roadmaps with implicit tensor-product search | Unbiased exploration by uniform sampling|
| **PRM\*** (`planners/prm_main_v2.py`) | Explicit composite configuration-space roadmap | Uniform sampling |


## Repo layout

```text
.
├── env/                         # Planning environments and task models
├── planners/                    # dRRT*, original dRRT*, PRM*, and ulities
├── models/                      # Robot/object assets and URDF resources
├── replay_saved_path.py         # Replay and visualize a saved planned path
├── analyze_first_solution_statistics.py
└── plot_trial_results.py        # Mean/95% CI convergence and path plots
```

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The code uses PyBullet for simulation and collision checking. On Linux, a graphical replay may require a working X/ OpenGL display; headless planning can generally be run with the GUI disabled in the selected environment.

## Running planners

```bash
# Proposed guided dRRT*
python3 -m planners.drrt_star_main

# Unbiased/original dRRT* baseline
python3 -m planners.drrt_star_original

# PRM* baseline
python3 -m planners.prm_main_v2
```

All the setup of environment factory, number of robots/objects, roadmap sample counts, transition sample counts, random seed range, iteration/time budget, and collision-checking setting are in these files.


## Environments


- `HandoverEnv` — cooperative object handover between multiple robots.
- `SimplePickPlaceEnv` — general multi-robot pick-and-place tasks.
- `SmallBoxPickPlaceEnv` — pick-and-place tasks with objects placed in a small box workspace.
- `ShelfPickPlaceEnv` — pick-and-place tasks involving shelf geometry.
- `StackingObjectsEnv` — stacking cube-like objects into a vertical arrangement.
- `StackingCastleEnv` — four-object stacking task with cube and triangular roof-shaped objects.
- `MultipleRobotsPickPlaceEnv` — pick-and-place tasks with multiple robots, including four-robot scenarios.
- `ThreeRobotsPickPlaceEnv` — pick-and-place tasks with three robots.
- `SimpleMotionPlanningWithSubgoalEnv` — 2-D multi-robot motion planning with explicit object subgoals. 


## Experiment outputs

Planner runs can save trial statistics and planned paths. A typical result records:

- planner variant and environment;
- seed and success/failure;
- first-solution time `T_first`;
- first-solution cost `J_first`;
- final cost `J_final`;
- iteration-level convergence information;
- the saved orbit/configuration path for replay.

### First-solution statistics

```bash
python3 analyze_first_solution_statistics.py \
  path/to/trial_statistics_seeds.txt
```

This reports per-seed first-solution values together with aggregate statistics and confidence intervals for successful runs.

### Simulate a saved path

```bash
python3 -m replay_saved_path \
  --path_file path/to/planned_path.txt \
  --num_robots 2 \
  --num_mobjects 2 \
  --output replay.mp4
```

Select the environment inside `replay_saved_path.py` when replaying a path from a different scenario. The path file must have been generated with the same environment definition and compatible robot/object counts.

<!--
MEDIA PLACEHOLDER — replay video
Add videos later, for example:

<video controls width="900">
  <source src="docs/media/example_replay.mp4" type="video/mp4">
</video>

Or link to an externally hosted video:
[Watch an example replay](PASTE_VIDEO_LINK_HERE)
-->


## Citation

```bibtex
@article{duong2026asymptotically,
  title={Asymptotically Optimal Multi-Robot Task and Motion Planning},
  author={Duong, Thi Thuy Ngan and Yiu, Cheuk Tung Shadow and Shome, Rahul and Sung, Yoonchang},
  journal={arXiv preprint arXiv:2609.18813},
  year={2026}
}
```

## License

Add the project license and any model/environment asset licenses here.

