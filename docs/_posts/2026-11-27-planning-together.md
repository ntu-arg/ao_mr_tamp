---
layout: post
title: "Planning together: from robot tasks to coordinated motion"
description: "An introduction to the multi-robot task and motion planning framework in this repository."
author: AO-MR-TAMP team
date: 2026-11-27 09:00:00 +0800
permalink: /planning-together.html
---

Moving an object from a table to a shelf sounds like a single instruction. For a robot, it opens up a series of decisions: which grasp to use, where to move, and how to avoid nearby obstacles. Add another robot and the problem becomes more involved. The robots must coordinate their motions, share workspace, and sometimes pass objects between them.

This repository explores **multi-robot task and motion planning (MR-TAMP)**: connecting discrete manipulation decisions with continuous, collision-free robot motion. Its central idea is to search for coordinated solutions using individual robot roadmaps, while continuing to explore task transitions and improve the best solution found so far.

## Why task and motion belong together

A task plan might say “pick up the object, hand it over, and place it on the shelf.” Each step still needs a feasible physical realization. A handover requires both robots to reach compatible configurations at the same time. A placement requires a reachable pose and a collision-free approach.

These decisions interact. A grasp that makes picking easy may make a later handover difficult. A route that works for one robot alone may interfere with another robot carrying an object. Planning tasks and motions together lets the search account for these dependencies.

## Two structures behind the search

The framework connects two kinds of structure:

- **Task modes and transitions.** A mode describes a manipulation arrangement, such as which robot holds an object. Transitions connect arrangements through actions such as picking, placing, or handing over. The planner also tracks orbits associated with these modes to organize exploration of feasible motion regions.
- **Individual robot roadmaps.** A roadmap records sampled configurations and candidate motions for a robot. Combining these roadmaps describes coordinated configurations of all robots.

Explicitly building every combination can become expensive as the team grows. The guided dRRT* planner instead searches an **implicit tensor product** of the constituent roadmaps: it explores combinations as needed without first constructing the full composite roadmap.

## Guidance with continued exploration

The implementation uses orbit, modal-state, frontier, and transition guidance to direct search toward useful parts of the problem. Once a solution exists, its structure can also guide further exploration.

Guidance helps decide where to spend computation, but continued exploration remains essential. Different manipulation transitions can constrain different subsets of robots. Improving motion inside one mode alone does not ensure that the search discovers a better sequence of transitions.

The project's asymptotic-optimality argument therefore depends on both persistent coverage of relevant transitions and asymptotically improving motion planning within connected feasible regions. This is a statement about behavior as planning resources grow under the required assumptions; it does not promise an optimal solution within a fixed runtime.

## Watch coordinated manipulation

The project demonstration shows the kind of multi-robot manipulation the framework targets.

<video controls playsinline preload="metadata" aria-label="Multi-robot manipulation demonstration">
  <source src="assets/ao_mrtamp.mp4" type="video/mp4">
  <a href="assets/ao_mrtamp.mp4">Download the demonstration video</a>.
</video>

## Explore the implementation

The repository includes three planner entry points for comparison:

<div class="table-scroll" markdown="1">

| Planner | Representation | Search guidance |
| --- | --- | --- |
| Guided dRRT* | Implicit tensor product of constituent roadmaps | Orbit, modal-state, frontier, and transition guidance |
| Original dRRT* | Implicit tensor product of constituent roadmaps | Uniform sampling |
| PRM* | Explicit composite configuration-space roadmap | Uniform sampling |

</div>

After installing the dependencies described in the [repository README](https://github.com/ntu-arg/ao_mr_tamp#installation), run the guided planner from the repository root:

```bash
python3 -m planners.drrt_star_main
```

The planner entry files contain the experiment settings, including the environment, robot and object counts, sampling budgets, seeds, and time limits. Available environments include pick-and-place, cooperative handover, shelf manipulation, and stacking tasks.

For experiments, distinguish **finding a first feasible solution** from **improving its cost**. The repository provides utilities to analyze first-solution statistics, plot convergence, and replay saved paths. Comparing both measures helps explain how a planner behaves under a finite computation budget.

## Where to go next

Start with the [project page](./) for the research overview and citation, then explore the [source code and setup instructions](https://github.com/ntu-arg/ao_mr_tamp). A useful first experiment is to select a pick-and-place environment, run the planner, and replay its saved path using the same environment and robot/object counts.

The larger question is how to make coordinated manipulation practical while retaining a principled route toward better solutions. This framework approaches that question by combining structured task exploration with motion search over reusable individual robot roadmaps.
