import sys


import pybullet as p
import pybullet_data
import numpy as np
import copy
import random
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


from env.simple_pick_place import SimplePickPlaceEnv, FrankaPandaRobot, MovableObject, FR3_LIMITS, \
     get_all_subsets, sample_yaw_quaternion, \
     make_transform
from planners.prm_roadmap_utils import Configuration, RobotState, ObjectState



NUM_TRIAL_SAMPLES = 10
GRASP_OFFSET_X = 0.0    # distance from end-effector to object when grasping, along the vertical direction (z-axis in EE frame)
GRASP_OFFSET_Y = 0.0
GRASP_OFFSET_Z = 0.025
IK_POSITION_TOLERANCE = 0.01  # meters

SAFE_APPROACH_DISTANCE = 0.015              # for collision checking: all parts should keep 1,5cm distance from each other for safe 
ALLOWED_PLACEMENT_CONTACT_DISTANCE = 0.025   # allow small contact distance for placed objects with the table (for stability)

# rgba colors for objects (for visualization)
cube_colors = [
                [1.0, 0.2, 0.2, 1.0],   # red
                [0.2, 0.6, 1.0, 1.0],   # blue
                [0.2, 0.9, 0.3, 1.0],   # green
                [1.0, 0.7, 0.2, 1.0],   # orange
            ]
class MovableObject:
    def __init__(self, init_pos, init_orientation, goal_regions=None, id=None, sim_id=None, using_client=None):
        self.x = init_pos[0]
        self.y = init_pos[1]
        self.z = init_pos[2]

        self.goal_regions = copy.deepcopy(goal_regions) if goal_regions is not None else None

        self.id = id                    # simple object id
        self.sim_id = sim_id            # simulator id (if needed for tracking in the planner)
        self.using_client = using_client      # pybullet client id for simulator interactions
        self.size = 0.05
        self.bounding_radius = 0.5 * np.sqrt(3.0) * self.size

        if self.sim_id is None:
            # Create pybullet object (cube) here and set self.sim_id
            obj_pos = [self.x, self.y, self.z]
            obj = p.loadURDF("cube_small.urdf", obj_pos, init_orientation, physicsClientId=self.using_client)
            p.changeVisualShape(obj, -1, rgbaColor=cube_colors[id % len(cube_colors)], physicsClientId=self.using_client)
            self.sim_id = obj

    def set_pose(self, t_object):
        """Set the object's pose in the simulator"""
        pos = t_object[:3, 3]
        rot = R.from_matrix(t_object[:3, :3])
        orn = rot.as_quat()
        p.resetBasePositionAndOrientation(self.sim_id, pos, orn, physicsClientId=self.using_client)

    def get_pose(self):
        """Get the object's current pose as a 4x4 homogeneous transformation matrix"""
        obj_state = p.getBasePositionAndOrientation(self.sim_id, physicsClientId=self.using_client)
        pos = obj_state[0]
        orn = obj_state[1]
        rot_matrix = p.getMatrixFromQuaternion(orn)
        t_object = np.eye(4)
        t_object[:3, :3] = np.array(rot_matrix).reshape(3, 3)
        t_object[:3, 3] = pos
        return t_object

    def get_position(self):
        """Get the object's current position as a 3D vector"""
        obj_state = p.getBasePositionAndOrientation(self.sim_id, physicsClientId=self.using_client)
        pos = obj_state[0]
        return np.array(pos)

    def get_goal_position(self):
        """Get the object's goal target as one region or list of regions."""
        return copy.deepcopy(self.goal_regions)

    def at_goal(self):
        if self.goal_regions is not None:
            for region in self.goal_regions:
                if all(region[axis][0] <= [self.x, self.y, self.z][axis] <= region[axis][1] for axis in range(3)):
                    return True
            return False
        if isinstance(self.x_goal, list) and len(self.x_goal) == 2:  # goal is a region defined by [min, max]
            return (
                self.x_goal[0] <= self.x <= self.x_goal[1]
                and self.y_goal[0] <= self.y <= self.y_goal[1]
                and self.z_goal[0] <= self.z <= self.z_goal[1]
            )
        return self.x == self.x_goal and self.y == self.y_goal and self.z == self.z_goal


class ShelfPickPlaceEnv(SimplePickPlaceEnv):

    " Robots pick objects from a small shelf and place them on the table."
    " Using robot: Franka Panda with 7 DOF and parallel gripper. "


    def __init__(self, num_robots=1, num_mobjects=1, show_gui=True):

        if show_gui:
            self.using_client = p.connect(p.GUI)
        else:
            self.using_client = p.connect(p.DIRECT)

        self.num_robots = num_robots
        self.num_mobjects = num_mobjects

        self.r_min = 1.0        # PRM lower bound connection threshold
        self.r_prm = 18.0       # PRM base scaling factor (gamma) for maximum initial search radius
        self.k_prm = 7          # PRM number of nearest neighbors to connect to

        # collision checking parameters
        # This is simple environment so we use coarse values of about 10 degrees for each joint.
        self.collision_checking_resolution = 0.5  # radian, changind depends on the complexity of evironment.
        self.collision_joint_resolution = 0.2     # radian

        self.robot_list = list(range(num_robots))
        self.transition_robot_subsets = get_all_subsets(self.robot_list)

        self.robots = []
        self.mobjects = []
        self.fixed_objects = []
        self.plane = None

        self.init_robot_states = {}
        self.init_object_states = {}
        self.init_attachments = {}          # No attachments at the start
        self.init_grasp_poses = {}          # No grasping poses at the start
        self.init_placement_poses = {}      # Init position of objects can be used as placement poses

        self.config_dim = self.num_robots * 7    # 7 joint angles for each robot
        self.configuration_type_radian = True
        self.configuration_type_cartesian = False

        self.joint_limits = FR3_LIMITS
        self.debugger_attached = sys.gettrace() is not None

        self.reset(client_id=self.using_client)

    # ========================
    # RESET
    # ========================
    def reset(self, client_id=None):
        if client_id is None:
            client_id = self.using_client
        # Keep all subsequent physics queries/actions on the same world as the latest reset.
        self.using_client = client_id
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.using_client)
        p.resetSimulation(physicsClientId=self.using_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.using_client)

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.using_client)

        # Table
        table_pos = [0.0, 0.0, 0.0] 
        table = p.loadURDF(
            "models/custom_table/custom_table.urdf", 
            table_pos, 
            p.getQuaternionFromEuler([0, 0, np.pi/2]),
            useFixedBase=True,
            physicsClientId=self.using_client
        )
        self.fixed_objects = [table]
        table_height = 0.88  # from table URDF

        # Shelf
        shelf_pos = [0.3, 0.0, table_height]           # placed on the table
        shelf = p.loadURDF(
            "models/two_floor_shelf/two_floor_shelf.urdf",
            shelf_pos,
            p.getQuaternionFromEuler([0, 0, np.pi/2]),
            useFixedBase=True,
            physicsClientId=self.using_client
        )
        objects_z_offset = 0.3                          # where the objects are placed relative to the shelf
        self.fixed_objects.append(shelf)


        # Robot base — mounted on the table surface (table z + table height=0.625)
        self.robots = []
        table_surface_z = table_pos[2] + table_height + 0.02    # 2cm support base of our franka robot
        offset_y = [-0.5, 0.5]                                  # our exp have two robots
        for i in range(self.num_robots):
            rid = i
            offset_x = -0.2
            offset_y_i = offset_y[i % len(offset_y)]
            robot_base_pos = [table_pos[0] + offset_x, table_pos[1] + offset_y_i, table_surface_z]
            robot = FrankaPandaRobot(id=rid, base_pos=robot_base_pos, client_id=self.using_client)
            self.robots.append(robot)

            self.init_attachments[rid] = None                   # No attachments at the start 
            self.init_robot_states[rid] = RobotState.TRANSIT    # All robots start in TRANSIT state


        # Create multiple movable objects
        # Goal is two same-size regions on the two sides of the shelf.
        x_range = [table_pos[0] + 0.1, table_pos[0] + 0.3]      # width 0.20
        z_range = [table_pos[2] + table_height, table_pos[2] + table_height + 0.05]  # height 0.05
        goal_regions = [
            [x_range, [table_pos[1] - 0.5, table_pos[1] - 0.3], z_range],
            [x_range, [table_pos[1] + 0.3, table_pos[1] + 0.5], z_range],
        ]
        self.goal_regions = goal_regions
        self.mobjects = []
        offset_y = [-0.06, 0.06]   # to place multiple objects on the shelf with some offset
        for i in range(self.num_mobjects):
            oid = i
            offset_x = -0.05             
            offset_y_i = offset_y[i % len(offset_y)]
            offset_z = objects_z_offset + 0.025	            # where to put the object and cube center at +0.025
            obj_position = [shelf_pos[0] + offset_x, shelf_pos[1] + offset_y_i, shelf_pos[2] + offset_z]
            obj_orientation = p.getQuaternionFromEuler([0, 0, 0])   # no rotation

            obj = MovableObject(
                id=oid, 
                using_client=self.using_client, 
                init_pos=obj_position, 
                init_orientation=obj_orientation,
                goal_regions=self.goal_regions
            )
            self.mobjects.append(obj)

            # init pose
            self.init_object_states[oid] = ObjectState.TRANSIT
            # Temporarily store the spawn pose; overwritten after physics settles.
            obj_pose = np.eye(4)
            obj_pose[:3, 3] = obj_position
            obj_pose[:3, :3] = np.array(p.getMatrixFromQuaternion(obj_orientation)).reshape(3, 3)
            self.init_placement_poses[oid] = obj_pose


        # Let newly spawned objects settle under gravity, then store the true stable poses.
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.using_client)

        for obj in self.mobjects:
            settled_pos, settled_orn = p.getBasePositionAndOrientation(
                obj.sim_id, physicsClientId=self.using_client
            )
            obj.x, obj.y, obj.z = [round(val, 3) for val in settled_pos]

            obj_pose = np.eye(4)
            obj_pose[:3, 3] = np.array([obj.x, obj.y, obj.z])
            obj_pose[:3, :3] = np.array(p.getMatrixFromQuaternion(settled_orn)).reshape(3, 3)
            self.init_placement_poses[obj.id] = obj_pose

            print("Created one movable object with id {}, initial position {}, and goal regions {}".format(
                obj.id, [obj.x, obj.y, obj.z], self.goal_regions
            ))

        self._draw_object_targets(using_client=self.using_client)  # Visualize target objects in the same client


    def _get_goal_conditions(self):
        
        """Return the goal conditions for the current environment, including target robot poses, object poses, and attachments."""
        robot_poses = []
        object_positions = []
        attachments = []  # List of tuples (robot_id, object_id) indicating which robot should be holding which object at the goal

        for _ in self.mobjects:
            object_positions.append(self.goal_regions)
        for robot in self.robots:
            robot_poses.append(robot.init_joint_angles)
            attachments.append(None) 		
        return robot_poses, object_positions, attachments
    
    def _extract_position_from_pose(self, pose):
        """Return xyz from either a 4x4 transform or a flat/vector pose-like container."""
        if pose is None:
            return None

        pose_arr = np.asarray(pose, dtype=float)
        if pose_arr.shape == (4, 4):
            return pose_arr[:3, 3]

        flat_pose = pose_arr.reshape(-1)
        if flat_pose.size >= 3:
            return flat_pose[:3]

        raise ValueError("pose must provide at least xyz or be a 4x4 transform")


    def is_goal_orbit(self, orbit):
        """Check if the given orbit satisfies the goal conditions of the current environment."""
        robot_goal_poses, object_goal_positions, goal_attachments = self._get_goal_conditions()

        # objects should be placed at goal
        for obj_id, target_pos in enumerate(object_goal_positions):
            current_pose = orbit.placement_poses.get(obj_id)
            current_position = self._extract_position_from_pose(current_pose)
            if current_position is None:
                return False  # object is not placed at all

            # Two-region goal format:
            # [ [[x_min, x_max], [y_min, y_max], [z_min, z_max]],
            #   [[x_min, x_max], [y_min, y_max], [z_min, z_max]] ]
            is_multi_region_goal = (
                isinstance(target_pos, (list, tuple, np.ndarray))
                and len(target_pos) > 0
                and all(
                    isinstance(region, (list, tuple, np.ndarray))
                    and len(region) == 3
                    and all(
                        isinstance(axis_range, (list, tuple, np.ndarray)) and len(axis_range) == 2
                        for axis_range in region
                    )
                    for region in target_pos
                )
            )

            if is_multi_region_goal:
                in_any_region = False
                for region in target_pos:
                    if all(region[axis_idx][0] <= current_position[axis_idx] <= region[axis_idx][1] for axis_idx in range(3)):
                        in_any_region = True
                        break
                if not in_any_region:
                    return False
                continue

            # Region goal format: [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
            is_region_goal = (
                isinstance(target_pos, (list, tuple, np.ndarray))
                and len(target_pos) == 3
                and all(
                    isinstance(axis_range, (list, tuple, np.ndarray)) and len(axis_range) == 2
                    for axis_range in target_pos
                )
            )

            if is_region_goal:
                for axis_idx, axis_range in enumerate(target_pos):
                    if not (axis_range[0] <= current_position[axis_idx] <= axis_range[1]):
                        return False
            else:
                target_position = np.asarray(target_pos, dtype=float)
                if not np.allclose(current_position, target_position, atol=0.05):  # allow some tolerance
                    return False
                
        # robot - holding object should be same as goal attachments
        for robot_id, holding_object in orbit.attachments.items():
            if holding_object != goal_attachments[robot_id]:
                return False
        
        orbit.is_goal_value = True
        return True
    

    def sample_boundary_configuration(self, orbit):
        '''
            Sample single configuration that lies on the boundary of this orbit.

            joint_angles: list of joint angles for all robots
                            joint angles is sampled based on IK for grasp and place, and random sample for robot not performing transition.

            pre_pose: is the safety pose that the robot need to reach before and after the grasp/ place pose.
                        pre_pose is computed based on the grasp and place pose, with a small offset along the approach direction,
                        or same with the random joint_angles if the robot is not performing transition action.
            
            new_attachments: new attachments after the transition action, should be different with the current attachments in the orbit.
            
            new_grasp_poses: only for the robot that is performing pick action, the new grasp pose for the object that is being picked.
            
            new_placement_poses: only for the robot that is performing place action, the new placement pose for the object that is being placed.
        '''

        # choose the subset of robots \Re
        transition_robots_id = random.choice(self.transition_robot_subsets)

        # available objects for pick action
        available_objects_id = list(range(self.num_mobjects))
        for grasping_obj_id in orbit.attachments.values():
            if grasping_obj_id in available_objects_id:
                available_objects_id.remove(grasping_obj_id)

        # Sample configuration for all robots
        joint_angles = []
        pre_pose = []
        new_attachments = copy.deepcopy(orbit.attachments)
        new_grasp_poses = {}
        new_placement_poses = {}
        for r_id in self.robot_list:
            if r_id in transition_robots_id:   # this robot is doing transition action
                # --------------------------
                # PICK ACTION
                # --------------------------
                if orbit.attachments[r_id] is None:  
                    if len(available_objects_id) == 0:
                        # No available objects to pick, this robot cannot do pick action, skip pick but random a joint angle
                        joint_angles.append(self.random_joint_angles_for_robot(r_id))
                        continue
                    else:
                        obj_id = random.choice(available_objects_id)
                        r_joint_angles, r_joint_angles_pre, T_grasp = sample_grasp_pose(obj_id, self, r_id, orbit)
                        if T_grasp is None or r_joint_angles is None or r_joint_angles_pre is None:
                            # robot cannot find feasible grasp pose for this object, skip pick but random a joint angle
                            random_joint_angles = self.random_joint_angles_for_robot(r_id)
                            joint_angles.append(random_joint_angles)
                            pre_pose.append(random_joint_angles)
                            continue
                        r_joint_angles = np.round(r_joint_angles, 4).tolist()
                        r_joint_angles_pre = np.round(r_joint_angles_pre, 4).tolist()
                        # update
                        joint_angles.append(r_joint_angles)
                        pre_pose.append(r_joint_angles_pre)
                        new_grasp_poses[obj_id] = T_grasp
                        new_attachments[r_id] = obj_id
                        available_objects_id.remove(obj_id)
                else:   
                    # --------------------------
                    # PLACE ACTION
                    # --------------------------
                    obj_id = orbit.attachments[r_id]
                    selected_region = random.choice(self.goal_regions)
                    x_goal_range, y_goal_range, z_goal_range = selected_region
                    x_place = round(random.uniform(x_goal_range[0], x_goal_range[1]) if isinstance(x_goal_range, (list, tuple)) else x_goal_range, 1)
                    y_place = round(random.uniform(y_goal_range[0], y_goal_range[1]) if isinstance(y_goal_range, (list, tuple)) else y_goal_range, 1)
                    z_place = round(random.uniform(z_goal_range[0], z_goal_range[1]) if isinstance(z_goal_range, (list, tuple)) else z_goal_range, 1)
                    r_joint_angles, r_joint_angles_pre, T_object = sample_placement_pose(obj_id, orbit.grasp_poses[obj_id], 
                                                                        x_goal=x_place, 
                                                                        y_goal=y_place, 
                                                                        z_goal=z_place,
                                                                        environment=self, r_id=r_id, orbit=orbit)
                    if T_object is None or r_joint_angles is None or r_joint_angles_pre is None:
                        # skip place but random a joint angle bcs robot can keep holding
                        random_joint_angles = self.random_joint_angles_for_robot(r_id)
                        joint_angles.append(random_joint_angles)
                        pre_pose.append(random_joint_angles)
                        continue
                    else:
                        r_joint_angles = np.round(r_joint_angles, 4).tolist()
                        r_joint_angles_pre = np.round(r_joint_angles_pre, 4).tolist()
                        # update
                        joint_angles.append(r_joint_angles)
                        pre_pose.append(r_joint_angles_pre)
                        new_placement_poses[obj_id] = T_object
                        new_attachments[r_id] = None
            else:
                # this robot is not doing transition action, just sample random joint angles
                random_joint_angles = self.random_joint_angles_for_robot(r_id)
                joint_angles.append(random_joint_angles)
                pre_pose.append(random_joint_angles)

        if len(pre_pose) != self.num_robots or len(joint_angles) != self.num_robots:
            raise ValueError("Mismatch in number of robots and sampled joint angles or pre-poses.")

        # Check if this is feasible
        if len(new_grasp_poses) + len(new_placement_poses) == 0:
            return None   # no actual transition action sampled, skip
        
        sampled_configuration = Configuration(
            joint_angles=joint_angles,
            pre_pose=pre_pose,
            attachments=new_attachments,
            grasp_poses=new_grasp_poses,
            placement_poses=new_placement_poses,
            is_transition=True,
            parent_orbit=orbit
        )
        if not self.is_feasible_at_boundary(sampled_configuration, orbit):
            return None
        return sampled_configuration
    


    ####################################################################
	#======================= Visualization Utils =======================
	####################################################################
    def _draw_object_targets(self, using_client=None):
        """Visualize target object locations"""
        if using_client is None:
            using_client = self.visualization_client
        target_color = [0, 0, 0]  # RGB for PyBullet debug APIs
        goals = getattr(self, "goal_regions", [self.mobjects[0].get_goal_position()])
        print(f"Visualizing goal regions at: {goals}")

        def _draw_region_box(region):
            x_min, x_max = region[0]
            y_min, y_max = region[1]
            z_min, z_max = region[2]
            corners = [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ]
            line_pairs = [
                (0, 1), (0, 2), (0, 4),
                (1, 3), (1, 5),
                (2, 3), (2, 6),
                (3, 7),
                (4, 5), (4, 6),
                (5, 7),
                (6, 7)
            ]
            for start_idx, end_idx in line_pairs:
                p.addUserDebugLine(
                    lineFromXYZ=corners[start_idx],
                    lineToXYZ=corners[end_idx],
                    lineColorRGB=target_color[:3],
                    lineWidth=2,
                    lifeTime=0,
                    physicsClientId=using_client
                )

        for region in goals:
            _draw_region_box(region)


#########################################
# MOTION PLANNING UTILS
#########################################
def sample_grasp_pose(obj_id, environment, r_id, orbit, pre_pose_offset=0.1):
    success = False
    # print(f'Robot {r_id} is trying to grasp object {obj_id}')

    for _ in range(NUM_TRIAL_SAMPLES):

        grasp_pose = random.choice([0, 1])

        # TODO: test side grasp
        grasp_pose = 0

        if grasp_pose == 0:
            # SIDE GRASP: roll = pi, pitch = -pi/2, yaw = 0
            target_orn = R.from_euler(
                'xyz',
                [np.pi, -np.pi/2, 0]
            ).as_quat().tolist()
        else:
            # TOP GRASP: roll = pi, pitch = 0, yaw = random
            target_orn = sample_yaw_quaternion(roll_target=np.pi, pitch_target=0)

        # For compute grasping pose
        T_grasp = make_transform([GRASP_OFFSET_X, GRASP_OFFSET_Y, GRASP_OFFSET_Z], target_orn) # EE to object transform (object pose in EE frame)
        T_object = orbit.placement_poses[obj_id]
        T_ee_grasp = T_object @ np.linalg.inv(T_grasp)

        # pre grasping pose
        T_appoach = np.eye(4)
        T_appoach[:3, 3] = np.array([0, 0, -pre_pose_offset])  # approach along negative z-axis in EE frame
        T_ee_pre_grasp = T_ee_grasp @ T_appoach

        # IK
        r_joint_angles = environment.robots[r_id].inverse_kinematics_from_ee_pose(T_ee_grasp)
        r_joint_angles_pre = environment.robots[r_id].inverse_kinematics_from_ee_pose(T_ee_pre_grasp)

        if r_joint_angles is None or r_joint_angles_pre is None:
            continue
        if not environment.is_grasp_feasible(obj_id, r_id, r_joint_angles, orbit):         # Broadphase collision check
            continue
        if not environment.is_pre_pose_feasible(r_id, r_joint_angles_pre, r_joint_angles, orbit):
            continue
        success = True
        break
    
    if success:
        return r_joint_angles, r_joint_angles_pre, T_grasp
    else:
        return None, None, None


def sample_placement_pose(obj_id, T_grasp, x_goal, y_goal, z_goal, environment, r_id, orbit=None, pre_pose_offset=0.1):

    success = False        
    z_offset = 0.02                             # add a small offset to prevent collision with the table

    for _ in range(NUM_TRIAL_SAMPLES):
        # side grasp but top placement
        target_orn = sample_yaw_quaternion(roll_target=0, pitch_target=np.pi/2, discretization=2)  # top placement

        # calculate placement pose
        T_obj_placement = make_transform(           # object pose in world frame
            [x_goal, y_goal, z_goal + z_offset],
            target_orn
        )
        T_ee_place = T_obj_placement @ np.linalg.inv(T_grasp)
        r_joint_angles = environment.robots[r_id].inverse_kinematics_from_ee_pose(T_ee_place)

        # calculate pre-place pose
        T_approach = np.eye(4)
        T_approach[:3, 3] = np.array([0, 0, -pre_pose_offset])  # approach along negative z-axis in EE frame
        T_ee_pre_place = T_ee_place @ T_approach
        r_joint_angles_pre = environment.robots[r_id].inverse_kinematics_from_ee_pose(T_ee_pre_place)

        if r_joint_angles is None or r_joint_angles_pre is None:
            continue
        if not environment.is_placement_collision_free(obj_id, r_id, r_joint_angles, orbit):         # Broadphase collision check
            continue
        if not environment.is_pre_pose_feasible(r_id, r_joint_angles_pre, r_joint_angles, orbit):     
            continue
        success = True
        break

    if success:
        T_object = make_transform([x_goal, y_goal, z_goal + z_offset], target_orn)
        return r_joint_angles, r_joint_angles_pre, T_object
    else:
        return None, None, None



if __name__ == "__main__":
    env = ShelfPickPlaceEnv(num_robots=2, num_mobjects=2)
    obs = env.reset()
    try:
        env.keep_gui_alive()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if sys.gettrace() is None:
            env.close()
        else:
            print("Debugger mode: leaving clients connected on exit.")
