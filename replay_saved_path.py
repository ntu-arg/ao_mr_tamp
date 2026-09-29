import argparse
import inspect
from pathlib import Path


from env.simple_pick_place import SimplePickPlaceEnv
from env.small_box_pick_place import SmallBoxPickPlaceEnv
from env.shelf_pick_place import ShelfPickPlaceEnv
from env.handover_object import HandoverEnv
from env.stacking_object import StackingObjectsEnv
from env.stacking_castle_objects import StackingCastleEnv
from env.simple_motion_planning import SimpleMotionPlanningWithSubgoalEnv
from env.four_robots_pick_place import MultipleRobotsPickPlaceEnv
from env.three_robots_pick_place import ThreeRobotsPickPlaceEnv


def main():
    parser = argparse.ArgumentParser(
        description="Replay a saved planned path text file and render simulation video."
    )
    parser.add_argument("--path_file", required=True, help="Input planned path txt file")
    parser.add_argument("--output", default="replay_from_saved_path.mp4", help="Output video filename")
    parser.add_argument("--delay", type=float, default=0.1, help="Per-step delay during replay")
    parser.add_argument(
        "--transition_pause",
        type=float,
        default=0.5,
        help="Seconds to hold each grasp/place transition so ordered actions are visible",
    )
    parser.add_argument("--num_robots", type=int, default=1, help="Number of robots in environment")
    parser.add_argument("--num_mobjects", type=int, default=1, help="Number of movable objects in environment")
    parser.add_argument(
        "--print_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print parsed path details before replay (default: enabled)",
    )
    args = parser.parse_args()

    path_file = Path(args.path_file).resolve()
    if not path_file.exists():
        raise FileNotFoundError(f"Path file not found: {path_file}")

    # env = SimplePickPlaceEnv(num_robots=args.num_robots, num_mobjects=args.num_mobjects, show_gui=True)
    # env = ShelfPickPlaceEnv(num_robots=args.num_robots, num_mobjects=args.num_mobjects)
    # env = SmallBoxPickPlaceEnv(num_robots=args.num_robots, num_mobjects=args.num_mobjects)
    # env = HandoverEnv(num_robots=args.num_robots, num_mobjects=args.num_mobjects)
    # env = StackingObjectsEnv(num_robots=2, num_mobjects=args.num_mobjects)
    # env = StackingCastleEnv(num_robots=2, num_mobjects=args.num_mobjects, show_gui=True)
    env = MultipleRobotsPickPlaceEnv(num_robots=4, num_mobjects=args.num_mobjects, show_gui=True)
    # env = ThreeRobotsPickPlaceEnv(num_robots=3, num_mobjects=args.num_mobjects, show_gui=True)
    # env = SimpleMotionPlanningWithSubgoalEnv(num_robots=args.num_robots, num_obstacles=4, num_mobjects=args.num_mobjects)


    # env = SharedBottleneck2DEnv()
    # env = AlternativeModalSequences2DEnv()
    # env = SameModalStateMultipleOrbits2DEnv()
    # env = CombinedModalOrbitChoice2DEnv()


    # Replay helpers are not uniform across environments.  Pass only options
    # supported by the selected environment instead of turning a replay-only
    # option into a hard failure.
    replay_kwargs = {
        "output_sim_filename": args.output,
        "delay": args.delay,
        "num_interpolate_points": 20,
    }
    if "transition_pause" in inspect.signature(env.simulate_final_path).parameters:
        replay_kwargs["transition_pause"] = args.transition_pause
    else:
        print(
            "Note: this environment does not support --transition_pause; "
            "using its built-in transition handling."
        )

    output = env.simulate_final_path(str(path_file), **replay_kwargs)
    print(f"Simulation output: {output}")


if __name__ == "__main__":
    main()
