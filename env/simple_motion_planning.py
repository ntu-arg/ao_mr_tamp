
import ast
import os
import re
import numpy as np
import random
import copy
import matplotlib

# Use Tk backend when available; otherwise fall back to non-interactive Agg.
try:
    import tkinter  # noqa: F401
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

from planners.prm_roadmap_utils import Configuration, RobotState, ObjectState


class Object:
    def __init__(self, x, y, xgoal, ygoal, id=None):
        self.x = x
        self.y = y
        self.xgoal = xgoal
        self.ygoal = ygoal
        self.id = id

    def get_goal_position(self):
        # In this simple environment, object goals are encoded by staying attached,
        # so use current position for visualization compatibility.
        return np.array([self.xgoal, self.ygoal])
    


class Robot:
    def __init__(self, x, y, id=None):
        self.x_start = x
        self.y_start = y
        self.id = id

        self.radius = 0.2    # simple circular robot with radius 0.2



class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self):
        plt.gca().add_patch(plt.Rectangle((self.x, self.y), self.width, self.height, color='gray', alpha=0.5))



class SimpleMotionPlanningWithSubgoalEnv():
    def __init__(self, num_robots=1, num_obstacles=4, num_mobjects=4):

        self.num_robots = num_robots
        self.num_obstacles = num_obstacles
        self.num_mobjects = num_mobjects

        self.r_min = 0.5        # PRM lower bound connection threshold
        self.r_prm = 8.0        # PRM base scaling factor (gamma) for maximum initial search radius
        self.k_prm = 7          # PRM number of nearest neighbors to connect to

        self.robot_list = list(range(num_robots))
        self.transition_robot_subsets = get_all_subsets(self.robot_list)

        self.robots = []
        self.mobjects = []
        self.obstacles = []

        self.start_states = {mobj_id: 'at_start' for mobj_id in range(num_mobjects)}    # initialize object states
        self.goal_states = {mobj_id: 'at_goal' for mobj_id in range(num_mobjects)}      # initialize goal states

        self.config_dim = num_robots * 2            # (x,y) for each robot
        self.configuration_type_radian = False
        self.configuration_type_cartesian = True

        # Build initial robots/objects/obstacles and symbolic init state.
        self.reset()

    
    def reset(self):

        # Keep reset idempotent when called multiple times.
        self.robots = []
        self.mobjects = []
        self.obstacles = []

        self.init_robot_states = {}
        self.init_object_states = {}
        self.init_attachments = {}          # No attachments at the start
        self.init_grasp_poses = {}          # No grasping poses at the start
        self.init_placement_poses = {}      # Init position of objects can be used as placement poses

        for i in range(self.num_robots):
            r_start = (0, i*3)
            self.robots.append(Robot(r_start[0], r_start[1], id=i))

            self.init_attachments[i] = None                   # No attachments at the start 
            self.init_robot_states[i] = RobotState.TRANSIT    # All robots start in TRANSIT state

        # Place objects in horizontal robot lanes.  Each goal is to the right
        # of its object, so a robot can move monotonically in +x:
        # start -> object -> goal -> next object -> next goal.
        objectx = [7, 7, 12, 12]
        objecty = [2, 5, 3, 0]
        goalx = [10, 10, 15, 15]
        goaly = [0, 3, 6, 5]
        
        for i in range(self.num_mobjects):
            o_x = objectx[i % len(objectx)]
            o_y = objecty[i % len(objecty)]
            o_xgoal = goalx[i % len(goalx)]
            o_ygoal = goaly[i % len(goaly)]
            self.mobjects.append(
                Object(
                    o_x,
                    o_y,
                    o_xgoal,
                    o_ygoal,
                    id=i,
                )
            )

            self.init_placement_poses[i] = (o_x, o_y)

        # Stagger obstacles in the gaps between horizontal lanes.  Obstacles
        # are kept away from the robot lanes and object start/goal points, so
        # the flat reference motions remain available.
        obstacle_templates = (
            (4.0, 1.0, 2.0, 1.0),
            (9.0, 1.2, 2.0, 1.0),
            (6.0, 3.0, 2.0, 1.0),
            (11.0, 4.2, 2.0, 1.0),
            (8.0, 7.2, 2.0, 1.0),
            (13.0, 7.2, 2.0, 1.0),
        )
        for i in range(self.num_obstacles):
            if i < len(obstacle_templates):
                obs_x, obs_y, obs_width, obs_height = obstacle_templates[i]
            else:
                # Continue the same staggered pattern for larger instances.
                row = i // 2
                obs_x = 4.0 if i % 2 == 0 else 9.0
                obs_y = 1.2 + row * 3.0
                obs_width = 2.0
                obs_height = 1.0
            self.obstacles.append(Obstacle(obs_x, obs_y, obs_width, obs_height))

        # set the limits of the environment
        self.x_min = -1
        maximum_object_x = max(obj.x for obj in self.mobjects)
        self.x_max = max(20, maximum_object_x + 2)
        self.y_min = -1
        maximum_lane_y = max(obj.y for obj in self.mobjects)
        maximum_goal_y = max(obj.ygoal for obj in self.mobjects)
        maximum_robot_radius = max(
            (robot.radius for robot in self.robots),
            default=0.0,
        )
        # Robot poses denote the robot center. Keep every object goal at least
        # one robot radius inside the workspace, with a small extra margin, so
        # a robot can physically reach the pose required to place it there.
        goal_workspace_margin = 0.2
        self.y_max = max(
            6,
            maximum_lane_y + 2,
            maximum_goal_y + maximum_robot_radius + goal_workspace_margin,
        )

        # Print the complete geometry so every experiment can be checked
        # against the exact environment instance used by the planner.
        print("\n=== SimpleMotionPlanningWithSubgoalEnv geometry ===")
        print(
            "Workspace bounds: "
            f"x=[{self.x_min}, {self.x_max}], "
            f"y=[{self.y_min}, {self.y_max}]"
        )
        print("Robots:")
        for robot in self.robots:
            print(
                f"  Robot {robot.id}: "
                f"({robot.x_start:.3f}, {robot.y_start:.3f})"
            )

        print("Objects:")
        for mobj in self.mobjects:
            print(
                f"  Object {mobj.id}: "
                f"init=({mobj.x:.3f}, {mobj.y:.3f}), "
                f"goal=({mobj.xgoal:.3f}, {mobj.ygoal:.3f})"
            )

        print("Obstacles (counter-clockwise vertices):")
        for obstacle_id, obstacle in enumerate(self.obstacles):
            x0, y0 = obstacle.x, obstacle.y
            x1, y1 = x0 + obstacle.width, y0 + obstacle.height
            vertices = (
                (x0, y0),
                (x1, y0),
                (x1, y1),
                (x0, y1),
            )
            formatted_vertices = ", ".join(
                f"({x:.3f}, {y:.3f})" for x, y in vertices
            )
            print(f"  Obstacle {obstacle_id}: {formatted_vertices}")
        
        print("=====================================================")
        print(
            "Environment initialized with "
            f"{len(self.robots)} robots, {len(self.mobjects)} objects, "
            f"and {len(self.obstacles)} obstacles."
        )
        print("Maximum number of orbits = 128")
        print("Optimal solution: 35.859")
        print("R0: (0,0) -> (7,2) -> (10,0) -> (12,0) -> (15,5) = 18.72")
        print("R1: (0,3) -> (7, 5) -> (10, 3) -> (12, 3) -> (13, 4.2) -> (15, 6) = 17.139")
        print(
            "Note: the orbit count and reference solution above are for the "
            "original 4-object layout."
        )
        print("=====================================================\n")

    
    def _get_goal_conditions(self):
        '''
            Get the goal conditions for this environment. 
            The goal condition is defined by the desired object states and attachments.
        '''
        object_positions = {}       # object has to be at specific position, if not object can be anywhere
        robot_poses = []            # a specific robot joint angles, if empty robot can take any valid configuration
        attachments = {}            # requirement if any robot is holding the object

        for mobj in self.mobjects:
            object_positions[mobj.id] = (mobj.xgoal, mobj.ygoal)

        for r_id, robot in enumerate(self.robots):
            o_id = r_id % self.num_mobjects
            attachments[robot.id] = None
            # robot should be at the same position as the object it is holding
            robot_poses.append((object_positions[o_id][0], object_positions[o_id][1]))

        return robot_poses, object_positions, attachments

    
    def is_goal_orbit(self, orbit) -> bool:
        '''
            Check if the given orbit satisfies the goal condition of this environment.
        '''

        # Goal: every robot is attached to its designated goal object. And object is at the goal
        robot_poses, object_positions, goal_attachments = self._get_goal_conditions()

        for obj_id, target_pos in object_positions.items():
            # Get current placement from orbit
            current_pose = orbit.placement_poses.get(obj_id)
            if current_pose is None:
                return False
            if not np.allclose(current_pose, target_pos, atol=0.1): # 0.1cm tolerance
                return False

        for r_id, goal_obj_id in goal_attachments.items():
            if orbit.attachments.get(r_id) != goal_obj_id:
                return False

        orbit.is_goal_value = True
        return True

    def get_object_goal_status(self, orbit):
        """Return per-object 2-D placement progress from the orbit state."""
        _, object_goal_positions, _ = self._get_goal_conditions()
        attached_object_ids = {
            object_id
            for object_id in orbit.attachments.values()
            if object_id is not None
        }
        status = []
        for movable_object in sorted(self.mobjects, key=lambda obj: obj.id):
            placement = orbit.placement_poses.get(movable_object.id)
            status.append(
                movable_object.id not in attached_object_ids
                and placement is not None
                and bool(
                    np.allclose(
                        np.asarray(placement, dtype=float).reshape(-1)[:2],
                        object_goal_positions[movable_object.id],
                        atol=0.1,
                    )
                )
            )
        return tuple(status)


    def get_init_configuration(self):
        '''
            Return joint angles of all robots in array of shape (num_robots, 2).
        '''
        robot_pose = []
        for i, robot in enumerate(self.robots):
            robot_pose.append([robot.x_start, robot.y_start])
        return np.array(robot_pose)


    def random_pose_for_robot(self, robot_id):
        '''
            Sample random pose for a specific robot within environment limits.
            No need to check feasibility.

            Returns: list of values [x, y] for the given robot_id
        '''
        x = round(random.uniform(self.x_min, self.x_max), 2)
        y = round(random.uniform(self.y_min, self.y_max), 2)
        return [x, y]


    def sample_interior_configuration(self, orbit):
        '''
            Sample a random configuration that lies in this orbit.
        '''

        # sample robot poses
        robot_poses = []
        for robot in self.robots:
            robot_poses.append(self.random_pose_for_robot(robot.id))

        # check of this is feasible
        sampled_config = Configuration(
            joint_angles=robot_poses,
            is_transition=False,
            attachments=orbit.attachments,
            parent_orbit=orbit
        )
        if not self.is_feasible_configuration(sampled_config, orbit):
            return None
        return sampled_config


    def sample_boundary_configuration(self, orbit):
        '''
             Sample single configuration that lies on the boundary of this orbit.
        '''

        # choose subset of robots to change attachment
        transition_robot_subset = random.choice(self.transition_robot_subsets)

        # available objects to be reached
        available_objects_id = list(range(self.num_mobjects))
        for grasping_obj_id in orbit.attachments.values():
            if grasping_obj_id in available_objects_id:
                available_objects_id.remove(grasping_obj_id)

        # This task is monotonic: once an object is placed at its goal, leave
        # it there.  Otherwise the sampler can create redundant pick/place
        # cycles and many extra geometric orbits for already completed objects.
        for object_id in list(available_objects_id):
            current_position = orbit.placement_poses.get(object_id)
            goal_position = self.mobjects[object_id].get_goal_position()
            if current_position is not None and np.allclose(
                np.asarray(current_position, dtype=float).reshape(-1)[:2],
                np.asarray(goal_position, dtype=float).reshape(-1)[:2],
                atol=0.1,
            ):
                available_objects_id.remove(object_id)

        # sample configuration for all robots
        robot_poses = []
        new_attachments = copy.deepcopy(orbit.attachments)
        new_grasp_poses = {}
        new_placement_poses = {}

        for r_id in self.robot_list:
            if r_id in transition_robot_subset: # this robot will change attachment, sample new attachment
                # ROBOT PICKING
                if orbit.attachments.get(r_id) is None:
                    if len(available_objects_id) == 0:
                        robot_poses.append(self.random_pose_for_robot(r_id))
                        continue
                    else:
                        new_obj_id = random.choice(available_objects_id)
                        object_position = orbit.placement_poses.get(new_obj_id)
                        grasp_pose = [object_position[0], object_position[1]]

                        robot_poses.append([object_position[0], object_position[1]])
                        new_grasp_poses[new_obj_id] = grasp_pose
                        new_attachments[r_id] = new_obj_id
                        available_objects_id.remove(new_obj_id)
                # ROBOT PLACING: placement pose will be object goal
                else:
                    placed_obj_id = orbit.attachments[r_id]
                    placement_pose = self.mobjects[placed_obj_id].get_goal_position()
                    new_placement_poses[placed_obj_id] = placement_pose
                    robot_poses.append([placement_pose[0], placement_pose[1]])
                    new_attachments[r_id] = None

            else: # this robot will not change attachment, keep the same attachment and sample new pose
                robot_poses.append(self.random_pose_for_robot(r_id))

        if len(robot_poses) != self.num_robots:
            return None

        # check of this is feasible
        sampled_config = Configuration(
            joint_angles=robot_poses,
            is_transition=True,
            attachments=new_attachments,
            grasp_poses=new_grasp_poses,
            placement_poses=new_placement_poses,
            # This environment has no separate pre-grasp/pre-place motion;
            # the sampled 2-D pose is already the motion endpoint expected by
            # the dRRT* constituent roadmap.
            pre_pose=robot_poses,
            parent_orbit=orbit
        )
        if not self.is_feasible_configuration(sampled_config, orbit):
            return None
        return sampled_config


    

    def is_feasible_configuration(self, config, orbit):
        """
            Check if the given configuration is collision free in the context of the given orbit 
            (considering attachments, grasp/placement poses)
        """

        poses = [self._as_robot_pose(pose) for pose in config.joint_angles]
        if len(poses) != self.num_robots:
            return False

        # Check robot-obstacle and workspace-boundary collisions.
        for robot_id, robot_pose in enumerate(poses):
            if not self._robot_pose_is_feasible(robot_id, robot_pose):
                return False

        # Level-2 checking for this 2-D environment includes robot-robot
        # separation.  This is intentionally done here, rather than in the
        # planner, so all planners use the same environment collision model.
        for first in range(self.num_robots):
            for second in range(first + 1, self.num_robots):
                minimum_distance = (
                    self.robots[first].radius + self.robots[second].radius
                )
                if np.linalg.norm(poses[first] - poses[second]) < minimum_distance:
                    return False

        return True


    @staticmethod
    def _as_robot_pose(pose):
        """Convert a 2-D robot pose to a validated numeric array."""
        values = np.asarray(pose, dtype=float).reshape(-1)
        if values.size != 2:
            raise ValueError(
                f"Expected a 2-D robot pose, got shape {np.asarray(pose).shape}"
            )
        return values


    def _robot_pose_is_feasible(self, robot_id, pose):
        """Check one robot against the workspace and static obstacles."""
        pose = self._as_robot_pose(pose)
        robot = self.robots[robot_id]
        x, y = pose
        radius = robot.radius

        if not (
            self.x_min + radius <= x <= self.x_max - radius
            and self.y_min + radius <= y <= self.y_max - radius
        ):
            return False

        for obstacle in self.obstacles:
            if (
                obstacle.x - radius <= x <= obstacle.x + obstacle.width + radius
                and obstacle.y - radius <= y <= obstacle.y + obstacle.height + radius
            ):
                return False
        return True


    @staticmethod
    def _node_robot_pose(node):
        """Extract the motion endpoint used by a constituent roadmap node."""
        if getattr(node, "is_transition", False):
            pre_pose = getattr(node, "pre_pose", None)
            if pre_pose is not None:
                return pre_pose
        return node.joint_angles


    def constituent_node_collision_checking(self, robot_id, q_sample, orbit=None):
        """dRRT* Level-1 node check for one 2-D robot."""
        if not 0 <= robot_id < self.num_robots:
            raise ValueError(f"Unknown robot id: {robot_id}")
        return self._robot_pose_is_feasible(robot_id, q_sample)


    def constituent_motion_collision_checking(
        self,
        robot_id,
        node_start,
        node_goal,
        orbit=None,
        resolution=None,
    ):
        """dRRT* Level-1 straight-line check for one 2-D robot."""
        if not 0 <= robot_id < self.num_robots:
            raise ValueError(f"Unknown robot id: {robot_id}")

        start = self._as_robot_pose(self._node_robot_pose(node_start))
        goal = self._as_robot_pose(self._node_robot_pose(node_goal))
        step = 0.05 if resolution is None else float(resolution)
        if step <= 0.0:
            raise ValueError("resolution must be > 0")

        checks = max(1, int(np.ceil(np.linalg.norm(goal - start) / step)))
        for index in range(checks + 1):
            pose = start + (goal - start) * (index / checks)
            if not self._robot_pose_is_feasible(robot_id, pose):
                return False
        return True


    def tensor_product_node_collision_checking(
        self,
        composite_joint_angles,
        orbit=None,
    ):
        """dRRT* Level-2 node check for all robots simultaneously."""
        joint_angles = getattr(composite_joint_angles, "joint_angles", composite_joint_angles)
        attachments = (
            dict(orbit.attachments)
            if orbit is not None
            else dict(self.init_attachments)
        )
        config = Configuration(
            joint_angles=np.asarray(joint_angles, dtype=float),
            is_transition=False,
            attachments=attachments,
            parent_orbit=orbit,
        )
        return self.is_feasible_configuration(config, orbit)


    def tensor_product_motion_collision_checking(
        self,
        start,
        goal,
        orbit=None,
        resolution=None,
    ):
        """dRRT* Level-2 straight-line check for the composite motion."""
        start_angles = getattr(start, "joint_angles", start)
        goal_angles = getattr(goal, "joint_angles", goal)
        start_array = np.asarray(start_angles, dtype=float)
        goal_array = np.asarray(goal_angles, dtype=float)
        expected_shape = (self.num_robots, 2)
        if start_array.shape != expected_shape or goal_array.shape != expected_shape:
            raise ValueError(
                f"Expected tensor configurations with shape {expected_shape}, "
                f"got {start_array.shape} and {goal_array.shape}"
            )

        step = 0.05 if resolution is None else float(resolution)
        if step <= 0.0:
            raise ValueError("resolution must be > 0")
        checks = max(1, int(np.ceil(np.linalg.norm(goal_array - start_array) / step)))
        for index in range(checks + 1):
            joint_angles = start_array + (goal_array - start_array) * (index / checks)
            if not self.tensor_product_node_collision_checking(joint_angles, orbit):
                return False
        return True



    def collision_free_motion(self, config1, config2, orbit=None, resolution=0.05, num_checks=None):
        """Check straight-line motion using adaptive midpoint checks until neighboring joint samples are within resolution."""
        if num_checks is None:
            num_checks = int(np.ceil(1 / resolution))

        for i in range(num_checks + 1):
            alpha = i / num_checks
            interpolated_joint_angles = (1 - alpha) * np.array(config1.joint_angles) + alpha * np.array(config2.joint_angles)
            interpolated_config = Configuration(
                joint_angles=interpolated_joint_angles,
                is_transition=False,  # intermediate configurations are not transitions
                attachments=config1.attachments,      
                parent_orbit=orbit,
            )
            if not self.is_feasible_configuration(interpolated_config, orbit):
                return False

        return True
    


    def close(self):
        return


    ############################################################
    #               VISUALIZATION AND UTILS FUNCTIONS
    ############################################################

    def simulate_final_path(self, output_path_filename, output_sim_filename="simulation.png", delay=0.2, num_interpolate_points=10):
        '''
            Simulate the final path and save to image file.
        '''

        if output_sim_filename.endswith(".mp4"):
            output_sim_filename = output_sim_filename[:-4] + ".png"

        def _parse_array(text):
            cleaned = text.strip()
            if not cleaned:
                return None
            try:
                return ast.literal_eval(cleaned)
            except Exception:
                # ``save_path_to_file`` may serialize NumPy arrays as
                # ``[7 5]`` rather than Python lists such as ``[7, 5]``.
                # Also restrict parsing to the first array in the block;
                # otherwise a following ``Pre Poses`` section can be
                # accidentally consumed as part of the current value.
                array_match = re.search(r"\[[^\]]*\]", cleaned, flags=re.S)
                array_text = array_match.group(0) if array_match else cleaned
                values = np.fromstring(
                    array_text.replace('[', ' ').replace(']', ' '),
                    sep=' ',
                )
                if values.size == 0:
                    return None
                return values.tolist()

        def _parse_pose_to_xy(pose):
            if pose is None:
                return None

            pose_array = np.asarray(pose, dtype=float)
            if pose_array.shape == (4, 4):
                return pose_array[:2, 3]

            flattened = pose_array.reshape(-1)
            if flattened.size >= 2:
                return flattened[:2]

            return None

        def _parse_configuration_section(section_text):
            joint_match = re.search(r"(?m)^\s*Joint Angles: (.+)$", section_text)
            if not joint_match:
                return None

            joint_angles = ast.literal_eval(joint_match.group(1).strip())

            attachments_match = re.search(r"(?m)^\s*Attachments: (.+)$", section_text)
            attachments = ast.literal_eval(attachments_match.group(1).strip()) if attachments_match else {}

            is_transition_match = re.search(r"(?m)^\s*Is Transition: (True|False)$", section_text)
            is_transition = is_transition_match.group(1) == "True" if is_transition_match else False

            grasp_poses = {}
            placement_poses = {}

            def _extract_named_blocks(block_header):
                header_match = re.search(rf"(?ms)^\s*{block_header}:[^\n]*\n(.*?)(?=^\s*(?:Grasp Poses|Placement Poses|Pre Poses|Parent Orbit IDs|Attachments|Is Transition|Shape|Joint Angles|q\d+ \(Configuration)|\Z)", section_text)
                if not header_match:
                    return {}

                block_text = header_match.group(1)
                parsed_items = {}
                for obj_id, value_block in re.findall(r"(?ms)^\s*Object\s+(\d+):\n(.*?)(?=^\s*Object\s+\d+:|\Z)", block_text):
                    parsed_value = _parse_array(value_block)
                    if parsed_value is not None:
                        parsed_items[int(obj_id)] = parsed_value
                return parsed_items

            grasp_poses.update(_extract_named_blocks("Grasp Poses"))
            placement_poses.update(_extract_named_blocks("Placement Poses"))

            return Configuration(
                joint_angles=joint_angles,
                attachments=attachments,
                is_transition=is_transition,
                grasp_poses=grasp_poses if is_transition else None,
                placement_poses=placement_poses if is_transition else None,
            )

        def _load_path_from_file(path_file):
            with open(path_file, "r") as f:
                text = f.read()

            orbit_path = []

            orbit_blocks = re.findall(r"(?ms)^Orbit\d+:.*?(?=^Orbit\d+:|\Z)", text)
            if orbit_blocks:
                for block in orbit_blocks:
                    configs = []
                    for match in re.finditer(r"(?ms)^\s*q\d+ \(Configuration \d+\):\n(.*?)(?=^\s*q\d+ \(Configuration \d+\):|\Z)", block):
                        cfg = _parse_configuration_section(match.group(1))
                        if cfg is not None:
                            configs.append(cfg)

                    if configs:
                        attachments_match = re.search(r"(?m)^\s*Attachments: (\{.*\})", block)
                        attachments = ast.literal_eval(attachments_match.group(1).strip()) if attachments_match else configs[0].attachments

                        orbit_path.append({
                            "configs": configs,
                            "attachments": attachments,
                        })

            if not orbit_path:
                sections = re.split(r"(?m)^--- Configuration \d+ ---\n", text)
                flat_configs = []
                for sec in sections[1:]:
                    cfg = _parse_configuration_section(sec)
                    if cfg is not None:
                        flat_configs.append(cfg)

                if flat_configs:
                    orbit_path = [{
                        "configs": flat_configs,
                        "attachments": flat_configs[0].attachments if flat_configs[0].attachments else {},
                    }]

            return orbit_path

        def _robot_xy_from_joint_angles(joint_angles):
            robot_xy = []
            for robot_pose in joint_angles:
                pose_array = np.asarray(robot_pose, dtype=float)
                robot_xy.append(pose_array[:2])
            return robot_xy

        def _object_positions_from_config(attachments, grasp_poses, placement_poses, robot_xy, current_object_positions):
            new_positions = {obj_id: np.array(pos, dtype=float) for obj_id, pos in current_object_positions.items()}

            for obj_id, pose in (placement_poses or {}).items():
                xy = _parse_pose_to_xy(pose)
                if xy is not None:
                    new_positions[int(obj_id)] = np.array(xy, dtype=float)

            for robot_id, held_obj_id in (attachments or {}).items():
                if held_obj_id is None:
                    continue

                attached_pose = None
                if grasp_poses and held_obj_id in grasp_poses:
                    attached_pose = grasp_poses[held_obj_id]

                offset_xy = np.zeros(2, dtype=float)
                if attached_pose is not None:
                    attached_array = np.asarray(attached_pose, dtype=float)
                    if attached_array.shape == (4, 4):
                        offset_xy = attached_array[:2, 3]
                    else:
                        flattened = attached_array.reshape(-1)
                        if flattened.size >= 2:
                            offset_xy = flattened[:2]

                new_positions[int(held_obj_id)] = np.array(robot_xy[int(robot_id)]) + offset_xy

            return new_positions

        orbit_path = _load_path_from_file(str(output_path_filename))
        total_configs = sum(len(segment.get("configs", [])) for segment in orbit_path)
        if total_configs == 0:
            print("No path configurations found in the input file.")
            return None

        output_path = os.path.abspath(output_sim_filename)
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        robot_trajectories = {robot.id: [] for robot in self.robots}
        object_trajectories = {mobj.id: [] for mobj in self.mobjects}
        object_positions = {mobj.id: np.array([mobj.x, mobj.y], dtype=float) for mobj in self.mobjects}

        for segment in orbit_path:
            configs = segment.get("configs", [])
            if not configs:
                continue

            for idx in range(len(configs) - 1):
                config1 = configs[idx]
                config2 = configs[idx + 1]
                num_steps = max(1, int(num_interpolate_points))

                for step in range(num_steps + 1):
                    alpha = step / num_steps
                    interpolated_joint_angles = (1 - alpha) * np.asarray(config1.joint_angles, dtype=float) + alpha * np.asarray(config2.joint_angles, dtype=float)
                    robot_xy = _robot_xy_from_joint_angles(interpolated_joint_angles.tolist())
                    object_positions = _object_positions_from_config(config1.attachments, config1.grasp_poses, config1.placement_poses, robot_xy, object_positions)

                    for robot_id, xy in enumerate(robot_xy):
                        robot_trajectories[robot_id].append(np.array(xy, dtype=float))
                    for obj_id, xy in object_positions.items():
                        object_trajectories[obj_id].append(np.array(xy, dtype=float))

            last_config = configs[-1]
            robot_xy = _robot_xy_from_joint_angles(last_config.joint_angles)
            object_positions = _object_positions_from_config(last_config.attachments, last_config.grasp_poses, last_config.placement_poses, robot_xy, object_positions)
            for robot_id, xy in enumerate(robot_xy):
                robot_trajectories[robot_id].append(np.array(xy, dtype=float))
            for obj_id, xy in object_positions.items():
                object_trajectories[obj_id].append(np.array(xy, dtype=float))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Final Planned Path')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.25)

        x_margin = 1.0
        y_margin = 1.0
        ax.set_xlim(self.x_min - x_margin, self.x_max + x_margin)
        ax.set_ylim(self.y_min - y_margin, self.y_max + y_margin)

        # Obstacles.
        for obs in self.obstacles:
            ax.add_patch(
                Rectangle(
                    (obs.x, obs.y),
                    obs.width,
                    obs.height,
                    facecolor='gray',
                    edgecolor='black',
                    alpha=0.5,
                )
            )

        # Colors
        base_robot_colors = [[0.2, 0.6, 1.0, 1.0], [0.2, 0.9, 0.3, 1.0]]  # Blue, Green
        # Keep the replay colors consistent with draw(), with one distinct
        # color for each of the four objects.
        base_object_colors = [
            [0.839, 0.153, 0.157, 1.0],  # red
            [1.000, 0.498, 0.055, 1.0],  # orange
            [0.737, 0.741, 0.133, 1.0],  # olive/yellow
            [0.545, 0.337, 0.490, 1.0],  # brown
        ]
        robot_colors = [base_robot_colors[i % len(base_robot_colors)] for i in range(len(self.robots))]
        object_colors = [base_object_colors[i % len(base_object_colors)] for i in range(len(self.mobjects))]

        # Static object locations, matching draw(): a circle is the object
        # start/subgoal and a star is its final goal.  Object trajectories are
        # intentionally not plotted here; only robot paths are dynamic.
        for mobj in self.mobjects:
            object_color = object_colors[mobj.id]
            ax.scatter(
                [mobj.x], [mobj.y], marker='o', s=120,
                c=[object_color], edgecolors='black', linewidths=0.6,
                zorder=5,
            )
            ax.text(
                mobj.x, mobj.y, f'O{mobj.id}', color='white', fontsize=8,
                ha='center', va='center', zorder=6,
            )
            ax.scatter(
                [mobj.xgoal], [mobj.ygoal], marker='*', s=150,
                c=[object_color], edgecolors='black', linewidths=0.6,
                zorder=5,
            )

        # Robot trajectories and final states.
        for robot_id, trajectory in robot_trajectories.items():
            if not trajectory:
                continue
            trajectory = np.asarray(trajectory, dtype=float)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=robot_colors[robot_id], linewidth=2, label=f'Robot {robot_id}')
            ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], color=robot_colors[robot_id], marker='o', s=60)
            ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], color=robot_colors[robot_id], marker='s', s=70)

        # Robot radii.
        # for robot in self.robots:
        #     ax.add_patch(
        #         Circle(
        #             (robot.x_start, robot.y_start),
        #             robot.radius,
        #             fill=False,
        #             edgecolor='black',
        #             linestyle=':',
        #             linewidth=1,
        #             alpha=0.5,
        #         )
        #     )

        legend_handles = [
            Line2D([0], [0], color=robot_colors[robot_id], linewidth=2,
                   label=f'Robot {robot_id} path')
            for robot_id in robot_trajectories
            if robot_trajectories[robot_id]
        ]
        for obj_id, mobj in enumerate(self.mobjects):
            object_color = object_colors[obj_id]
            legend_handles.extend([
                Line2D(
                    [0], [0], marker='o', linestyle='None',
                    markerfacecolor=object_color, markeredgecolor='black',
                    markersize=8, label=f'Object {obj_id} start (circle)',
                ),
                Line2D(
                    [0], [0], marker='*', linestyle='None',
                    markerfacecolor=object_color, markeredgecolor='black',
                    markersize=11, label=f'Object {obj_id} goal (star)',
                ),
            ])
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        print(f"Saved 2D final path image: {output_path}")
        return output_path
    

    def draw(self, show=True, output_path=None):
        '''
            Draw the environment with robots, objects, and obstacles.
        '''
        # Keep draw() usable even if reset() has not been called yet.
        if not self.robots or not self.mobjects or not self.obstacles:
            self.reset()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Environment')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.25)

        x_margin = 1.0
        y_margin = 1.0
        ax.set_xlim(self.x_min - x_margin, self.x_max + x_margin)
        ax.set_ylim(self.y_min - y_margin, self.y_max + y_margin)

        # Obstacles.
        for obs in self.obstacles:
            ax.add_patch(
                Rectangle(
                    (obs.x, obs.y),
                    obs.width,
                    obs.height,
                    facecolor='gray',
                    edgecolor='black',
                    alpha=0.5,
                )
            )

        # Use stable color cycling so different robots/objects remain distinguishable.
        robot_palette = ['#1f77b4', '#2ca02c', '#17becf', '#9467bd']
        object_palette = ['#d62728', '#ff7f0e', '#bcbd22', '#8c564b']

        # Robots.
        for robot in self.robots:
            robot_color = robot_palette[robot.id % len(robot_palette)]
            ax.add_patch(
                Circle(
                    (robot.x_start, robot.y_start),
                    robot.radius,
                    facecolor=robot_color,
                    edgecolor='black',
                    alpha=0.7,
                )
            )
            ax.text(robot.x_start, robot.y_start, f'R{robot.id}', color='white', fontsize=8, ha='center', va='center')

        # Objects.
        legend_handles = []

        for mobj in self.mobjects:
            object_color = object_palette[mobj.id % len(object_palette)]
            ax.scatter([mobj.x], [mobj.y], marker='o', s=120, c=object_color, edgecolors='black', linewidths=0.6, zorder=5)
            ax.text(mobj.x, mobj.y, f'O{mobj.id}', color='white', fontsize=8, ha='center', va='center')

            ax.scatter([mobj.xgoal], [mobj.ygoal], marker='*', s=150, c=object_color, edgecolors='black', linewidths=0.6, zorder=5)
            legend_handles.extend([
                Line2D([0], [0], marker='o', linestyle='None',
                       markerfacecolor=object_color, markeredgecolor='black',
                       markersize=8, label=f'Object {mobj.id} start (circle)'),
                Line2D([0], [0], marker='*', linestyle='None',
                       markerfacecolor=object_color, markeredgecolor='black',
                       markersize=11, label=f'Object {mobj.id} goal (star)'),
            ])

        legend_handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=robot_palette[robot.id % len(robot_palette)],
                   markeredgecolor='black', markersize=8,
                   label=f'Robot {robot.id}')
            for robot in self.robots
        ] + legend_handles
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)

        fig.tight_layout()

        backend_name = matplotlib.get_backend().lower()
        is_non_interactive_backend = 'agg' in backend_name
        if show and not is_non_interactive_backend:
            plt.show()
        elif show and is_non_interactive_backend:
            if output_path is None:
                output_path = os.path.abspath('simple_motion_planning_with_subgoal_env.png')
            fig.savefig(output_path, dpi=200)
            print(f"Non-interactive backend '{matplotlib.get_backend()}' detected. Saved figure to {output_path}")

        return fig, ax



def get_subset(set, k):
    '''
        Get all subsets of size k from the given set.
    '''
    if k == 0:
        return [[]]
    if not set:
        return []
    
    first = set[0]
    rest = set[1:]

    subsets_with_first = [[first] + subset for subset in get_subset(rest, k-1)]
    subsets_without_first = get_subset(rest, k)

    return subsets_with_first + subsets_without_first

def get_all_subsets(set):
    '''
        Get all subsets of the given set.
    '''
    all_subsets = []
    for k in range(1, len(set) + 1):
        all_subsets.extend(get_subset(set, k))
    return all_subsets

# test
if __name__ == "__main__":
    env = SimpleMotionPlanningWithSubgoalEnv(num_robots=2, num_obstacles=4, num_mobjects=4)
    # env.reset()
    env.draw()

    # test
    # print(get_all_subsets([0, 1, 2, 3, 4]))
