import os
import shutil
import ast
import re
import sys

import pybullet as p
import pybullet_data
import numpy as np
from collections import defaultdict
import time
import copy
import random
from scipy.spatial.transform import Rotation as R

from env.env_utils import get_all_subsets, interpolate_motion
from planners.prm_roadmap_utils import Configuration, RobotState, ObjectState



NUM_TRIAL_SAMPLES = 10
GRASP_OFFSET_X = 0.0    # distance from end-effector to object when grasping, along the vertical direction (z-axis in EE frame)
GRASP_OFFSET_Y = 0.0
GRASP_OFFSET_Z = 0.0
IK_POSITION_TOLERANCE = 0.01  # meters

SAFE_APPROACH_DISTANCE = 0.015              # for collision checking: all parts should keep 1,5cm distance from each other for safe 
ALLOWED_PLACEMENT_CONTACT_DISTANCE = 0.025   # allow small contact distance for placed objects with the table (for stability)

# rgba colors for objects (for visualization)
cube_colors = [
                [1.0, 0.2, 0.2, 1.0],
                [0.2, 0.6, 1.0, 1.0],
                [0.2, 0.9, 0.3, 1.0],
                [1.0, 0.7, 0.2, 1.0],
            ]

# Joint limits for the FR3 arm.
_deg = np.deg2rad
# FR3_LIMITS = (
#     (_deg(-166), _deg(166)),   # J1
#     (_deg(-105), _deg(105)),   # J2
#     (_deg(-166), _deg(166)),   # J3
#     (_deg(-176), _deg(-7)),    # J4
#     (_deg(-165), _deg(165)),   # J5
#     (_deg(25), _deg(265)),     # J6
#     (_deg(-175), _deg(175)),   # J7
# )
FR3_LIMITS = (                      # safer limits
    (_deg(-127.3), _deg(127.3)),   # J1
    (_deg(-81.7),  _deg(81.7)),    # J2
    (_deg(-137.9), _deg(137.9)),   # J3
    (_deg(-152.4), _deg(-30.6)),   # J4
    (_deg(-137.1), _deg(137.1)),   # J5
    (_deg(53.8),   _deg(236.2)),   # J6
    (_deg(-149.1), _deg(149.1)),   # J7
)

#------------------------------
#----- FOR DEBUGGING ONLY -----
print_debug = False
#----------- END ---------


class MovableObject:
    def __init__(self, init_pos, init_orientation, x_goal, y_goal, z_goal, id, sim_id=None, client_id=None):
        self.x = init_pos[0]
        self.y = init_pos[1]
        self.z = init_pos[2]    # from the center of the cube to the table surface
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.z_goal = z_goal
        self.id = id                    # simple object id
        self.sim_id = sim_id           # simulator id (if needed for tracking in the planner)
        self.client_id = client_id     # pybullet client id for simulator interactions
        self.size = 0.05               # overall width of the object, for collision checking
        self.bounding_radius = 0.5 * np.sqrt(3.0) * self.size

        if self.sim_id is None:
            # Create pybullet object (cube) here and set self.sim_id
            obj_pos = [self.x, self.y, self.z]
            obj = p.loadURDF("cube_small.urdf", obj_pos, init_orientation, physicsClientId=self.client_id)
            p.changeVisualShape(obj, -1, rgbaColor=cube_colors[id % len(cube_colors)], physicsClientId=self.client_id)
            self.sim_id = obj

        # Keep track of the pose most recently applied through set_pose().
        # Placed objects are repeatedly assigned the same orbit-level pose
        # during edge checking, and resetting an unchanged body needlessly
        # updates PyBullet's broadphase state.
        self._last_applied_pose = self.get_pose()

    def set_pose(self, t_object):
        """Restore the object's pose and clear residual dynamics."""
        pose = np.asarray(t_object, dtype=float)
        command_unchanged = np.allclose(
            pose,
            self._last_applied_pose,
            rtol=0.0,
            atol=1e-10,
        )
        if command_unchanged:
            # Physics may have moved a dynamic body since the last command.
            # The cache is only a hint; PyBullet's actual pose is authoritative.
            actual_pose = self.get_pose()
            if np.allclose(pose, actual_pose, rtol=0.0, atol=1e-9):
                p.resetBaseVelocity(
                    self.sim_id,
                    linearVelocity=[0.0, 0.0, 0.0],
                    angularVelocity=[0.0, 0.0, 0.0],
                    physicsClientId=self.client_id,
                )
                return False

        pos = pose[:3, 3]
        rot = R.from_matrix(pose[:3, :3])
        orn = rot.as_quat()
        p.resetBasePositionAndOrientation(self.sim_id, pos, orn, physicsClientId=self.client_id)
        p.resetBaseVelocity(
            self.sim_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )
        self._last_applied_pose = pose.copy()
        return True

    def get_pose(self):
        """Get the object's current pose as a 4x4 homogeneous transformation matrix"""
        obj_state = p.getBasePositionAndOrientation(self.sim_id, physicsClientId=self.client_id)
        pos = obj_state[0]
        orn = obj_state[1]
        rot_matrix = p.getMatrixFromQuaternion(orn)
        t_object = np.eye(4)
        t_object[:3, :3] = np.array(rot_matrix).reshape(3, 3)
        t_object[:3, 3] = pos
        return t_object

    def get_position(self):
        """Get the object's current position as a 3D vector"""
        obj_state = p.getBasePositionAndOrientation(self.sim_id, physicsClientId=self.client_id)
        pos = obj_state[0]
        return np.array(pos)
    
    def get_goal_position(self):
        """Get the object's goal position as a 3D vector"""
        return np.array([self.x_goal, self.y_goal, self.z_goal])

    def at_goal(self):
        return self.x == self.x_goal and self.y == self.y_goal and self.z == self.z_goal

    


class FrankaPandaRobot:
    def __init__(self, id, sim_id=None, base_pos=[0, 0, 0], client_id=None):
        self.id = id
        self.sim_id = sim_id
        self.client_id = client_id
        self.base_pos = base_pos
        self.init_joint_angles = [0, -0.5, 0, -2.5, 0, 2.0, 0.8]  # Neutral pose
        self.joint_limits = FR3_LIMITS

        if self.sim_id is None:
            # Create pybullet robot here and set self.sim_id
            self.sim_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=base_pos,
                useFixedBase=True,
                physicsClientId=self.client_id
            )

        # Testing panda link infor
        # for i in range(p.getNumJoints(self.sim_id)):
        #     info = p.getJointInfo(self.sim_id, i)
        #     print(f"Joint {i}: {info[1].decode()} | Link: {info[12].decode()}")

        self.max_reach = 1.06                               # maximum reach range for franka arm is 1060mm (including the gripper), this is for collision checking filter 

        self.movable_joint_indices = [0, 1, 2, 3, 4, 5, 6]  # Panda arm joints
        self.ee_link_index = 11
        self.fingertip_link_indices = [8, 9, 10]            # Panda gripper fingertip links
        self.base_link_index = [-1, 0]                      # allow contact with robot base and table (for support)
        self.allowed_contacts = {                           # for self-collision checking
            # Adjacent arm links
            (0,1), (1,0),
            (1,2), (2,1),
            (2,3), (3,2),
            (3,4), (4,3),
            (4,5), (5,4),
            (5,6), (6,5),
            (6,7), (7,6),

            # Wrist / hand
            (6,8), (8,6),
            (7,8), (8,7),

            # Fingers attached to hand
            (8,9), (9,8),
            (8,10), (10,8),

            # Finger internals
            (9,10), (10,9),

            # Sometimes needed due to Panda geometry
            (5,7), (7,5),

            # Allow contact between gripper and held object
            (8, 11), (11, 8),
            (9, 11), (11, 9),
            (10, 11), (11, 10),
        }

        # Reset joints to neutral pose
        for i in self.movable_joint_indices:
            p.resetJointState(self.sim_id, i, self.init_joint_angles[i], physicsClientId=self.client_id)

    def get_end_effector_pose(self):
        """Get the current end-effector pose as a 4x4 homogeneous transformation matrix"""
        link_state = p.getLinkState(
            self.sim_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        pos = link_state[0]
        orn = link_state[1]
        rot_matrix = p.getMatrixFromQuaternion(orn)
        t_ee = np.eye(4)
        t_ee[:3, :3] = np.array(rot_matrix).reshape(3, 3)
        t_ee[:3, 3] = pos
        return t_ee
    
    
    def forward_kinematics(self, joint_angles):
        """Compute the forward kinematics for the given joint angles and return the end-effector pose as a 4x4 homogeneous transformation matrix"""
        if len(joint_angles) != len(self.movable_joint_indices):
            raise ValueError(
                f"Expected {len(self.movable_joint_indices)} joint angles, got {len(joint_angles)}"
            )

        previous_joint_states = []
        for i in self.movable_joint_indices:
            joint_state = p.getJointState(self.sim_id, i, physicsClientId=self.client_id)
            previous_joint_states.append(joint_state[0])  # position

        # Set the robot to the given joint angles
        for joint_idx, angle in zip(self.movable_joint_indices, joint_angles):
            p.resetJointState(
                self.sim_id,
                joint_idx,
                angle,
                physicsClientId=self.client_id
            )
        # Get the resulting end-effector pose
        ee_pose = self.get_end_effector_pose()

        # Reset robot back to original pose (to avoid side effects)
        for i, angle in zip(self.movable_joint_indices, previous_joint_states):
            p.resetJointState(self.sim_id, i, angle, physicsClientId=self.client_id)

        return ee_pose


    def inverse_kinematics(self, target_pos, target_orn):
        """Compute the inverse kinematics for the given target end-effector pose and robot_id
        
            :param target_pos: Target end-effector position as a list of 3 values [x, y, z]
            :param target_orn: Target end-effector orientation as a quaternion [x, y, z, w]
            :param robot_id: ID of the robot to compute IK for
            :param environment: The environment object containing the robots
            :return: Joint angles as a list of 7 values, or None if no solution is found
        """
        
        robot_ee_link = self.ee_link_index
        movable_joint_indices = self.movable_joint_indices
        previous_joint_angles = [
            p.getJointState(
                self.sim_id,
                joint_idx,
                physicsClientId=self.client_id,
            )[0]
            for joint_idx in movable_joint_indices
        ]
        lower_limits = [lower for lower, _ in self.joint_limits]
        upper_limits = [upper for _, upper in self.joint_limits]
        joint_ranges = [
            upper - lower
            for lower, upper in self.joint_limits
        ]
        rest_poses = [
            float(np.clip(angle, lower, upper))
            for angle, (lower, upper) in zip(
                previous_joint_angles,
                self.joint_limits,
            )
        ]

        # Use PyBullet's built-in IK solver
        ik_solution = p.calculateInverseKinematics(
            self.sim_id,
            robot_ee_link,
            target_pos,                                 # position as list of 3 values [x, y, z]
            targetOrientation=target_orn,               # orientation as quaternion
            physicsClientId=self.client_id,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=200,
            residualThreshold=1e-4
        )

        if not ik_solution:
            return None

        joint_angles = list(ik_solution[:7])  # Return only the first 7 joint angles for the arm
        if any(
            angle < lower or angle > upper
            for angle, (lower, upper) in zip(
                joint_angles,
                self.joint_limits,
            )
        ):
            return None

        # PyBullet IK can return a numeric solution even when the target is unreachable.
        # Validate by checking the FK end-effector position error at the solved joints.
        for joint_idx, angle in zip(movable_joint_indices, joint_angles):
            p.resetJointState(
                self.sim_id,
                joint_idx,
                angle,
                physicsClientId=self.client_id
            )

        ee_state = p.getLinkState(
            self.sim_id,
            robot_ee_link,
            computeForwardKinematics=True,
            physicsClientId=self.client_id
        )
        achieved_pos = np.asarray(ee_state[0], dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        position_error = np.linalg.norm(achieved_pos - target_pos)

        # Restore previous robot joint state to avoid side effects in planning checks.
        for joint_idx, angle in zip(movable_joint_indices, previous_joint_angles):
            p.resetJointState(
                self.sim_id,
                joint_idx,
                angle,
                physicsClientId=self.client_id
            )
        if position_error > IK_POSITION_TOLERANCE:
            return None
        return joint_angles
    

    def inverse_kinematics_from_ee_pose(self, target_pose):
        """Compute the inverse kinematics for the given target end-effector pose and robot
        
            :param target_pose: Target end-effector pose as a 4x4 numpy homogeneous transformation matrix
            :return: Joint angles as a list of 7 values, or None if no solution is found
        """
        target_pos = target_pose[:3, 3]
        rot_matrix = target_pose[:3, :3]
        target_orn = rotation_matrix_to_quaternion(rot_matrix)

        return self.inverse_kinematics(target_pos, target_orn)
    

    def is_self_colliding(self, q_sample):
        """
        Set the robot to the sampled configuration and check for self-collisions.
        """

        for join_idx, angle in zip(self.movable_joint_indices, q_sample):
            p.resetJointState(self.sim_id, join_idx, angle, physicsClientId=self.client_id)
        
        contacts = p.getClosestPoints(self.sim_id, self.sim_id, distance=0.0, physicsClientId=self.client_id)
        
        # Filter out 'adjacent' joints
        for contact in contacts:
            link_a = contact[3]
            link_b = contact[4]
            if (link_a, link_b) in self.allowed_contacts or abs(link_a - link_b) <= 1:
                continue
            if print_debug:
                print(f"Self-collision detected for Robot{self.id} between link {link_a} and link {link_b}")
                print(f"Contact details: {contact}")
            return True
        return False


    def set_pose(self, joint_angles):
        """Set the robot's joint angles in the simulator"""
        for joint_idx, angle in zip(self.movable_joint_indices, joint_angles):
            p.resetJointState(self.sim_id, joint_idx, angle, physicsClientId=self.client_id)



class SimplePickPlaceEnv:
    """
        The environment of pick and place objects inside one table.
        Using robot: Franka Panda

        Searching space for robot: 
        [
            [joint1_r1, joint2_r1, ..., joint7_r1], # Robot 1 state
            [joint1_r2, joint2_r2, ..., joint7_r2]  # Robot 2 state
        ]

    """

    def __init__(self, num_robots=1, num_mobjects=1, show_gui=True):

        if show_gui:
            self.using_client = p.connect(p.GUI, options="--width=1600 --height=1000",)
        else:
            self.using_client = p.connect(p.DIRECT)

        self.num_robots = num_robots
        self.num_mobjects = num_mobjects

        # PRM parameters
        self.r_min = 1.0            # PRM lower bound connection threshold
        self.r_prm = 80.0           # PRM base scaling factor (gamma) for maximum initial search radius
        self.k_prm = 7              # PRM number of nearest neighbors to connect to

        # collision checking parameters
        # This is simple environment so we use coarse values of about 10 degrees for each joint.
        self.collision_checking_resolution = 0.5  # radian, changind depends on the complexity of evironment.
        self.collision_joint_resolution = 0.2     # radian


        self.robot_list = list(range(num_robots))
        self.transition_robot_subsets = get_all_subsets(self.robot_list)

        self.robots = []
        self.mobjects = []
        self.plane = None


        self.init_robot_states = {}
        self.init_object_states = {}
        self.init_attachments = {}          # No attachments at the start
        self.init_grasp_poses = {}          # No grasping poses at the start
        self.init_placement_poses = {}      # Init position of objects can be used as placement poses

        self.config_dim = self.num_robots * 7    # 7 joint angles for each robot

        self.joint_limits = FR3_LIMITS
        self.debugger_attached = sys.gettrace() is not None

        # Collision acceleration state is initialized lazily because derived
        # environments override reset() and create their own PyBullet bodies.
        self._collision_world_signature = None
        self._constituent_collision_contexts = {}
        self._tensor_collision_contexts = {}
        self._constituent_node_validity_cache = {}
        self._tensor_node_validity_cache = {}
        self._fixed_body_aabbs = {}

        self.reset(client_id=self.using_client)


    @staticmethod
    def _object_y_layout(num_objects):
        """Return safe initial/goal Y offsets for the rotated 1.6 m table.

        The custom table is rotated by 90 degrees when loaded, so its long
        dimension occupies world Y in [-0.8, 0.8]. Keep multi-object layouts
        within [-0.6, 0.6] to leave clearance for each cube at both ends.
        """
        if num_objects <= 0:
            return [], []
        if num_objects <= 2:
            initial = [0.2 - index * 0.4 for index in range(num_objects)]
            goals = [0.3 - index * 0.7 for index in range(num_objects)]
            return initial, goals

        positions = np.linspace(0.6, -0.6, num_objects).tolist()
        return positions, list(positions)



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
        table1_pos = [0.5, 0.0, 0.0]
        table = p.loadURDF(
            "models/custom_table/custom_table.urdf", 
            table1_pos, 
            p.getQuaternionFromEuler([0, 0, np.pi/2]),
            useFixedBase=True,
            physicsClientId=self.using_client
        )
        self.fixed_objects = [table]
        table_height = 0.88


        # Robot base is mounted 2 cm above the 0.88 m tabletop.
        self.robots = []
        table_surface_z = table1_pos[2] + table_height
        robot_mount_z = table_surface_z + 0.02
        offset_y = [0.5, -0.5]  
        for i in range(self.num_robots):
            rid = i                                             # Force to start from 0 to be consistent with the joint angles in the configuration array
            offset_y_i = offset_y[i % len(offset_y)]
            offset_x_i = -0.2
            robot_base_pos = [table1_pos[0] + offset_x_i, table1_pos[1] + offset_y_i, robot_mount_z]
            robot = FrankaPandaRobot(id=rid, base_pos=robot_base_pos, client_id=self.using_client)          
            self.robots.append(robot)

            self.init_attachments[rid] = None                   # No attachments at the start 
            self.init_robot_states[rid] = RobotState.TRANSIT    # All robots start in TRANSIT state


        # Create multiple movable objects
        self.mobjects = []
        initial_y_offsets, goal_y_offsets = self._object_y_layout(
            self.num_mobjects
        )
        for i in range(self.num_mobjects):
            oid = i
            offset_x = 0.15
            offset_y = initial_y_offsets[i]
            offset_z = 0.025   # from the center of the cube to the table surface
            obj_pos = [table1_pos[0] + offset_x, table1_pos[1] + offset_y, table_surface_z + offset_z]  # on the table surface
            obj_orientation = p.getQuaternionFromEuler([0, 0, 0])   # no rotation

            offset_x_goal = 0.3
            # offset_y_goal = 1*i
            # target_pos = [table1_pos[0] + offset_x_goal, table1_pos[1] - 0.5 + offset_y_goal, table_surface_z]    # target location

            offset_y_goal = goal_y_offsets[i]
            target_pos = [
                table1_pos[0] + offset_x_goal,
                table1_pos[1] + offset_y_goal,
                table_surface_z + offset_z,
            ]

            obj = MovableObject(
                obj_pos, 
                obj_orientation,
                x_goal=target_pos[0], y_goal=target_pos[1], z_goal=target_pos[2], 
                id=oid, sim_id=None,
                client_id=self.using_client
            )

            # init pose
            obj_pose = np.eye(4)
            obj_pose[:3, 3] = obj_pos
            obj_pose[:3, :3] = np.array(p.getMatrixFromQuaternion(obj_orientation)).reshape(3, 3)
            self.init_placement_poses[oid] = obj_pose    
            self.mobjects.append(obj)

            self.init_object_states[oid] = ObjectState.TRANSIT  # All objects start at being free

            print("Created one movable object with id {}, initial position {}, and goal position {}".format(
                oid, obj_pos, target_pos
            ))


        self._draw_object_targets(client_id=self.using_client)  # Visualize target objects in the same client
        
        # FOR CHECKING GRASPING POSES
        # state = p.getLinkState(self.robots[0].sim_id, self.robots[0].ee_link_index, physicsClientId=self.using_client)
        # pos = state[4]
        # orn = state[5]
        # print(f"EE pose: pos {pos}, orn {orn}")
        # self.draw_frame(pos, orn)

        # obj_state = p.getBasePositionAndOrientation(self.mobjects[0].sim_id, physicsClientId=self.using_client)
        # obj_pos = obj_state[0]
        # obj_orn = obj_state[1]
        # self.draw_frame(obj_pos, obj_orn)

        # Turn of all the coordinates
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def simulate_goal(self):
        """
            Simulate the goal state of the environment for visualization.
        """
        # Set all objects to their goal positions
        for mobject in self.mobjects:
            goal_pos = [mobject.x_goal, mobject.y_goal, mobject.z_goal]
            goal_orn = p.getQuaternionFromEuler([0, 0, 0])  # Assuming no rotation for simplicity
            goal_pose = np.eye(4)
            goal_pose[:3, :3] = np.asarray(p.getMatrixFromQuaternion(goal_orn)).reshape(3, 3)
            goal_pose[:3, 3] = goal_pos
            mobject.set_pose(goal_pose)

        # Set all robots to their initial joint angles (or any other desired configuration)
        for robot in self.robots:
            for joint_idx, angle in zip(robot.movable_joint_indices, robot.init_joint_angles):
                p.resetJointState(robot.sim_id, joint_idx, angle, physicsClientId=self.using_client)

        # Step the simulation to visualize the changes
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.using_client)
            time.sleep(1./240.)  # Adjust sleep time as needed for visualization


    def _get_goal_conditions(self):
        '''
            Read the goal conditions of the current environment
        '''
        # object is at goal placement pose and not held by any robot
        object_positions = {}       # object has to be at specific position, if not object can be anywhere
        robot_poses = []            # a specific robot joint angles, if empty robot can take any valid configuration
        attachments = {}            # requirement if any robot is holding the object

        for mobject in self.mobjects:
            object_positions[mobject.id] = (mobject.x_goal, mobject.y_goal, mobject.z_goal)
        for robot in self.robots:
            robot_poses.append(robot.init_joint_angles)
            attachments[robot.id] = None
        return robot_poses, object_positions, attachments
    
    
    def is_goal_orbit(self, orbit):
        '''
            Check if the given orbit satisfies the goal conditions of the current environment
        '''
        robot_goal_poses, object_goal_positions, goal_attachments = self._get_goal_conditions()

        for obj_id, target_pos in object_goal_positions.items():
            # Get current placement from orbit
            current_pose = orbit.placement_poses.get(obj_id)
            if current_pose is None:
                return False
            if not self._position_satisfies_goal_spec(
                current_pose[:3, 3],
                target_pos,
                atol=0.02,
            ):
                return False

        for r_id, target_obj_id in goal_attachments.items():
            actual_obj_id = orbit.attachments.get(r_id)
            if actual_obj_id != target_obj_id:
                return False

        orbit.is_goal_value = True
        return True

    @staticmethod
    def _position_satisfies_goal_spec(position, goal_spec, atol=0.02):
        """Evaluate a point goal, one box region, or several box regions."""
        if position is None:
            return False

        is_axis_range = lambda value: (
            isinstance(value, (list, tuple, np.ndarray))
            and len(value) == 2
        )
        is_region = lambda value: (
            isinstance(value, (list, tuple, np.ndarray))
            and len(value) == 3
            and all(is_axis_range(axis_range) for axis_range in value)
        )

        if is_region(goal_spec):
            return all(
                axis_range[0] <= position[axis] <= axis_range[1]
                for axis, axis_range in enumerate(goal_spec)
            )

        if (
            isinstance(goal_spec, (list, tuple, np.ndarray))
            and len(goal_spec) > 0
            and all(is_region(region) for region in goal_spec)
        ):
            return any(
                all(
                    axis_range[0] <= position[axis] <= axis_range[1]
                    for axis, axis_range in enumerate(region)
                )
                for region in goal_spec
            )

        return bool(
            np.allclose(
                position,
                np.asarray(goal_spec, dtype=float),
                atol=atol,
            )
        )

    def get_object_goal_status(self, orbit):
        """Return per-object task progress using only immutable orbit data."""
        _, object_goal_specs, _ = self._get_goal_conditions()
        attached_object_ids = {
            object_id
            for object_id in orbit.attachments.values()
            if object_id is not None
        }
        status = []
        for movable_object in sorted(self.mobjects, key=lambda obj: obj.id):
            if movable_object.id in attached_object_ids:
                status.append(False)
                continue

            placement_pose = orbit.placement_poses.get(movable_object.id)
            if placement_pose is None:
                status.append(False)
                continue

            pose_array = np.asarray(placement_pose, dtype=float)
            current_position = (
                pose_array[:3, 3]
                if pose_array.shape == (4, 4)
                else pose_array.reshape(-1)[:3]
            )
            goal_spec = object_goal_specs[movable_object.id]
            status.append(
                self._position_satisfies_goal_spec(
                    current_position,
                    goal_spec,
                    atol=0.02,
                )
            )
        return tuple(status)


    def get_init_configuration(self):
        '''
            Return joint angles of all robots in array of shape (num_robots, 7).
        '''
        joint_angles = []
        for i, robot in enumerate(self.robots):
            joint_angles.append(robot.init_joint_angles)
        return np.array(joint_angles)


    def random_joint_angles_for_robot(self, robot_id):
        '''
            Sample random joint angles for a specific robot within joint limits.
        '''
        robot = self.robots[robot_id]
        sampled_joint_angles = []
        for joint_idx in robot.movable_joint_indices:
            low, high = self.joint_limits[joint_idx]
            angle = round(float(random.uniform(low, high)), 4)
            sampled_joint_angles.append(angle)
        return sampled_joint_angles


    def sample_interior_configuration(self, orbit):
        '''
            Sample a random configuration that lies in this orbit.
        '''

        sampled_joint_angles = []
        for robot in self.robots:
            robot_joint_angles = self.random_joint_angles_for_robot(robot.id)
            sampled_joint_angles.append(robot_joint_angles)

        # Check if this is feasible
        sampled_configuration = Configuration(
            joint_angles=sampled_joint_angles,
            attachments=orbit.attachments,
            is_transition=False,
            parent_orbit=orbit
        )
        if not self.is_feasible_in_orbit(sampled_joint_angles, orbit):
            return None
        return sampled_configuration

    def sample_boundary_configuration(self, orbit):
        '''
            Sample single configuration that lies on the boundary of this orbit.

            joint_angles: list of joint angles for all robots
                            joint angles is sampled based on IK for grasp and place, 
                            and random sample for robot not performing transition.

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
        available_objects_id = [obj.id for obj in self.mobjects]
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
                        random_joint_angles = self.random_joint_angles_for_robot(r_id)
                        joint_angles.append(random_joint_angles)
                        pre_pose.append(random_joint_angles)
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
                        # round them up
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
                    x_place = self.mobjects[obj_id].x_goal  # FIXME: this is ideal case
                    y_place = self.mobjects[obj_id].y_goal
                    z_place = self.mobjects[obj_id].z_goal
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
                        # round them up
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

        # Check if this is feasible
        if len(new_grasp_poses) + len(new_placement_poses) == 0:
            return None   # no actual transition action sampled, skip
        
        if len(pre_pose) != self.num_robots or len(joint_angles) != self.num_robots:
            raise ValueError(" !!!! Mismatch in number of robots and sampled joint angles or pre-poses. Check env.sample_boundary_configuration() logic!!!")

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


    def is_self_colliding(self, robot_id, q_sample):
        """
        Set the robot to the sampled configuration and check for self-collisions.
        """
        return self.robots[robot_id].is_self_colliding(q_sample)


    def _has_forbidden_collision(self, all_robot_sim_ids, attachment_sim_ids, placed_objects_sim_ids, debug_context=""):
        """Run layered collision checks and return True when any forbidden collision is found."""
        # getClosestPoints performs a pair-specific collision query, so a
        # full-world performCollisionDetection pass is unnecessary here.

        robot_ids = list(all_robot_sim_ids)
        fixed_object_ids = list(self.fixed_objects)
        placed_ids = set(placed_objects_sim_ids)
        attached_pairs = {
            r_sim: o_sim
            for r_sim, o_sim in attachment_sim_ids.items()
            if o_sim is not None
        }
        attached_obj_ids = set(attached_pairs.values())
        movable_obj_ids = sorted(placed_ids | attached_obj_ids) # set union

        fingertip_links_by_robot = {
            robot.sim_id: set(robot.fingertip_link_indices)
            for robot in self.robots
        }

        def _debug_collision(contact, prefix="Collision detected"):
            if not print_debug:
                return
            bodyA = contact[1]
            bodyB = contact[2]
            contact_distance = contact[8]

            bodyA_name = None
            bodyB_name = None
            for r_id, robot in enumerate(self.robots):
                if bodyA == robot.sim_id:
                    bodyA_name = f"Robot{r_id}"
                if bodyB == robot.sim_id:
                    bodyB_name = f"Robot{r_id}"
            for o_id, obj in enumerate(self.mobjects):
                if bodyA == obj.sim_id:
                    bodyA_name = f"Object{o_id}"
                if bodyB == obj.sim_id:
                    bodyB_name = f"Object{o_id}"
            if bodyA in self.fixed_objects:
                bodyA_name = f"Table{bodyA}"
            if bodyB in self.fixed_objects:
                bodyB_name = f"Table{bodyB}"
            if bodyA == self.plane:
                bodyA_name = f"Plane{bodyA}"
            if bodyB == self.plane:
                bodyB_name = f"Plane{bodyB}"
            if bodyA_name is None:
                bodyA_name = f"Body{bodyA}"
            if bodyB_name is None:
                bodyB_name = f"Body{bodyB}"

            if debug_context:
                print(f"In {debug_context} checking:: {prefix} between {bodyA_name} and {bodyB_name} with contact distance {contact_distance}")
            else:
                print(f"{prefix} between {bodyA_name} and {bodyB_name} with contact distance {contact_distance}")
            print(f"Contact details: {contact}")

        def _first_penetration_contacts(body_a, body_b, max_distance=0.0):
            contacts = p.getClosestPoints(
                bodyA=body_a,
                bodyB=body_b,
                distance=max_distance,               # ignore points of distance > max_distance
                physicsClientId=self.using_client,
            )
            return [c for c in contacts if c[8] <= 0]

        # LAYER 2: ATTACHED OBJECTS INTERACTIONS
        # Layer 2a: attached object against its owning robot (allow fingertip links only).
        for robot_sim_id, attached_obj_sim_id in attached_pairs.items():
            for c in _first_penetration_contacts(robot_sim_id, attached_obj_sim_id):
                link_robot = c[3] if c[1] == robot_sim_id else c[4]
                if link_robot in fingertip_links_by_robot.get(robot_sim_id, set()):
                    continue
                _debug_collision(c, prefix="Forbidden attachment collision")
                return True

        # Layer 2b: movable objects (placed and attached) with fixed objects.
        for obj_sim_id in movable_obj_ids:
            for fixed_sim_id in fixed_object_ids:
                contacts = _first_penetration_contacts(obj_sim_id, fixed_sim_id)
                for c in contacts:
                    if obj_sim_id in placed_ids and c[8] >= -ALLOWED_PLACEMENT_CONTACT_DISTANCE:
                        continue
                    if c[8] <= 0:
                        _debug_collision(c, prefix="Object-environment collision")
                        return True
            # with plane
            for c in _first_penetration_contacts(obj_sim_id, self.plane):
                if obj_sim_id in placed_ids and c[8] >= -ALLOWED_PLACEMENT_CONTACT_DISTANCE:
                    continue
                if c[8] <= 0:
                    _debug_collision(c, prefix="Object-plane collision")
                    return True

        # Layer 2c: movable objects with each other.
        movable_obj_list = sorted(movable_obj_ids)
        for i in range(len(movable_obj_list)):
            for j in range(i + 1, len(movable_obj_list)):
                for c in _first_penetration_contacts(movable_obj_list[i], movable_obj_list[j], max_distance=0.0):
                    _debug_collision(c, prefix="Object-object collision")
                    return True


        # LAYER 3: ROBOT INTERACTIONS
        # Layer 3a: robot-robot collisions.
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                for c in _first_penetration_contacts(robot_ids[i], robot_ids[j]):
                    _debug_collision(c, prefix="Robot-robot collision")
                    return True

        # Layer 3b: robot with fixed objects, allowing base/table support contacts.
        for robot_sim_id in robot_ids:
            robot_base_links = set(self.robots[robot_ids.index(robot_sim_id)].base_link_index)
            for fixed_sim_id in fixed_object_ids:
                for c in _first_penetration_contacts(robot_sim_id, fixed_sim_id):
                    link_robot = c[3] if c[1] == robot_sim_id else c[4]
                    if link_robot in robot_base_links:   # TODO: avoid hardcoding the robot base link
                        continue
                    _debug_collision(c, prefix="Robot-environment collision")
                    return True
            # robot with plane
            for c in _first_penetration_contacts(robot_sim_id, self.plane):
                _debug_collision(c, prefix="Robot-plane collision")
                return True

        # Layer 3c: robot with movable objects.
        for robot_sim_id in robot_ids:
            owner_obj_sim_id = attached_pairs.get(robot_sim_id)
            for obj_sim_id in movable_obj_ids:
                for c in _first_penetration_contacts(robot_sim_id, obj_sim_id):
                    if owner_obj_sim_id == obj_sim_id:
                        link_robot = c[3] if c[1] == robot_sim_id else c[4]
                        if link_robot in fingertip_links_by_robot.get(robot_sim_id, set()):
                            continue
                    _debug_collision(c, prefix="Robot-object collision")
                    return True

        return False
    

    def is_placement_collision_free(self, object_id, robot_id, robot_joint_angles, orbit):
        """Check if the given placement is valid (no collisions)
        
            This placement collision free only checks placment pose of objects is collision free,
            other collision would computed later in the transition configuration feasibility check.
        
        """

        # Feasibility check
        # Robot self-collision check at the place pose
        if self.is_self_colliding(robot_id, robot_joint_angles):
            if print_debug:
                print(f"Place pose for Robot{robot_id} is in self-collision.")
            return False
        
        # Set gripper closed since robot is holding object during placement
        p.resetJointState(self.robots[robot_id].sim_id, 9, 0.0, physicsClientId=self.using_client)
        p.resetJointState(self.robots[robot_id].sim_id, 10, 0.0, physicsClientId=self.using_client)

        # Extract the values
        t_ee = self.robots[robot_id].forward_kinematics(robot_joint_angles)
        current_object_sim_id = self.mobjects[object_id].sim_id
        placed_objects_sim_ids = []
        for obj_id, placement_pose in orbit.placement_poses.items():
            placed_objects_sim_ids.append(self.mobjects[obj_id].sim_id)
        
        # Place all placed objects
        if orbit is not None:
            for obj_id, placement_pose in orbit.placement_poses.items():
                self.mobjects[obj_id].set_pose(
                    placement_pose
                )

        # Attach this object to robot EE
        holding_object = orbit.attachments[robot_id]
        if holding_object is not None:
            t_object = (
                t_ee
                @ orbit.grasp_poses[holding_object]
            )
            self.mobjects[holding_object].set_pose(                 # set placement action for this object
                t_object
            )

        
        # Collision checking between placing objs and other movable objects
        for obj_sim_id in placed_objects_sim_ids:
            contacts = p.getClosestPoints(obj_sim_id, current_object_sim_id, distance=0.0, physicsClientId=self.using_client)
            for c in contacts:
                if c[8] >= SAFE_APPROACH_DISTANCE:
                    continue
                return False
        # Collision checking between placing objects and fixed environment objects
        for fixed_sim_id in self.fixed_objects:
            contacts = p.getClosestPoints(current_object_sim_id, fixed_sim_id, distance=0.0, physicsClientId=self.using_client)
            for c in contacts:
                if c[8] >= SAFE_APPROACH_DISTANCE:
                    continue
                return False
        
        return True
    

    def is_grasp_feasible(self, object_id, robot_id, robot_joint_angles, orbit):
        """Check if the given grasp is feasible (no collisions)"""



        robot_sim_id = self.robots[robot_id].sim_id
        target_object_sim_id = self.mobjects[object_id].sim_id


        # set the object pose for the placed objects
        for obj_id, placement_pose in orbit.placement_poses.items():
            self.mobjects[obj_id].set_pose(
                placement_pose
            )


        # Feasibility check
        # Robot self-collision check at the grasp pose
        if self.is_self_colliding(robot_id, robot_joint_angles):
            if print_debug:
                print(f"Grasp pose for Robot{robot_id} is in self-collision.")
            return False
        
        # Set gripper open since robot hasn't grasped yet
        p.resetJointState(self.robots[robot_id].sim_id, 9, 0.04, physicsClientId=self.using_client)
        p.resetJointState(self.robots[robot_id].sim_id, 10, 0.04, physicsClientId=self.using_client)
        
        # Robot's hand doesn't smash into the environment
        base_links = set(self.robots[robot_id].base_link_index)
        for fixed_sim_id in self.fixed_objects:
            contacts = p.getClosestPoints(
                robot_sim_id,
                fixed_sim_id,
                distance=0.0,
                physicsClientId=self.using_client,
            )
            for c in contacts:
                if c[8] >= SAFE_APPROACH_DISTANCE:
                    continue
                robot_link = c[3] if c[1] == robot_sim_id else c[4]
                if robot_link in base_links:
                    continue
                if robot_link in self.robots[robot_id].fingertip_link_indices and c[8] >= SAFE_APPROACH_DISTANCE/10: # allow close contact with fingertip links
                    continue
                if print_debug:
                    print(f"Grasp pose for Robot{robot_id} has forbidden collision with environment object {fixed_sim_id}.")
                    print(f"Contact details: {c}")
                return False

        # Gripper with object collision check at the grasp pose (allow contact with fingertip links)
        # fingertip_links = set(self.robots[robot_id].fingertip_link_indices)
        # contacts = p.getClosestPoints(
        #     robot_sim_id,
        #     target_object_sim_id,
        #     distance=SAFE_APPROACH_DISTANCE,
        #     physicsClientId=self.using_client,
        # )
        # for c in contacts:
        #     if c[8] >= SAFE_APPROACH_DISTANCE:
        #         continue
        #     robot_link = c[3] if c[1] == robot_sim_id else c[4]
        #     if robot_link in fingertip_links and c[8] >= 0.0:       # allow contact with fingertip links, but not allow penetration
        #         continue
        #     return False

        return True
    

    def is_feasible_in_orbit(self, joint_angles, orbit):
        """Check if the given configuration is feasible in the context of the given orbit (considering attachments, grasp/placement poses)
        
            :param joint_angles: list of joint angles for all robots
            :param orbit: the orbit context to check feasibility against

            :return: True if feasible, False otherwise
        
        """

        # move robot to the test configuration
        t_ee_poses = {}
        for r_id in range(self.num_robots):
            q_sample = joint_angles[r_id]
            if self.is_self_colliding(r_id, q_sample):
                return False
            # is_self_colliding already reset the robot to q_sample.
            # Now set the gripper state based on attachment
            gripper_target = 0.0 if orbit.attachments.get(r_id) is not None else 0.04
            p.resetJointState(self.robots[r_id].sim_id, 9, gripper_target, physicsClientId=self.using_client)
            p.resetJointState(self.robots[r_id].sim_id, 10, gripper_target, physicsClientId=self.using_client)
            t_ee_poses[r_id] = self.robots[r_id].get_end_effector_pose()

        # move objects to the test configuration
        placed_objects = []
        placed_objects_sim_ids = []
        objects_placements = self._placed_object_poses_for_collision(orbit)
        if objects_placements is not None:
            for object_id, placement_pose in objects_placements.items():
                self.mobjects[object_id].set_pose(placement_pose)
                placed_objects.append(object_id)
                placed_objects_sim_ids.append(self.mobjects[object_id].sim_id)

        # attach the object to the robot (simulate the robot holding the object)
        for robot_id, holding_object in orbit.attachments.items():
            if holding_object is not None:
                t_ee = t_ee_poses[robot_id]  # get the current end-effector pose
                t_object = t_ee @ orbit.grasp_poses[holding_object]  # t_object = t_ee * t_grasp
                # print(f"Input config for Robot{robot_id} joint angles: {joint_angles[robot_id]}")
                # print(f"Testing attachment of Object{holding_object} to Robot{robot_id} at grasp pose:\n{orbit.grasp_poses[holding_object]}")
                # print(f"End-effector pose:\n{t_ee}")
                # print(f"Resulting object pose:\n{t_object}")
                self.mobjects[holding_object].set_pose(t_object)

        attachment_sim_ids = {}
        for r_id, holding_object in orbit.attachments.items():
            if holding_object is not None:
                r_sim_id = self.robots[r_id].sim_id
                obj_sim_id = self.mobjects[holding_object].sim_id
                attachment_sim_ids[r_sim_id] = obj_sim_id


        all_robot_sim_ids = [r.sim_id for r in self.robots]
        if self._has_forbidden_collision(
            all_robot_sim_ids,
            attachment_sim_ids,
            placed_objects_sim_ids,
            debug_context="interior orbit",
        ):
            return False
        return True


    def _placed_object_poses_for_collision(self, orbit):
        """Return placed movable objects that remain physical obstacles.

        Ordinary pick-and-place environments keep every placed object at its
        recorded orbit pose. Environments with a container may override this
        hook when a recorded placement represents an object that subsequently
        leaves the manipulation workspace, for example by falling into a box.
        """
        return orbit.placement_poses if orbit is not None else {}
    

    def is_feasible_at_boundary(self, configuration, orbit):
        """
        Check if the given configuration is feasible at the boundary of the given orbit (considering attachments, grasp/placement poses)
        """
        
        # Set robot to the test configuration
        t_ee_poses = {}
        for r_id in range(self.num_robots):
            q_sample = configuration.joint_angles[r_id]
            if self.is_self_colliding(r_id, q_sample):
                return False
            # is_self_colliding already reset the robot to q_sample.
            # Now set the gripper state based on attachment
            gripper_target = 0.0 if configuration.attachments.get(r_id) is not None else 0.04
            p.resetJointState(self.robots[r_id].sim_id, 9, gripper_target, physicsClientId=self.using_client)
            p.resetJointState(self.robots[r_id].sim_id, 10, gripper_target, physicsClientId=self.using_client)
            t_ee_poses[r_id] = self.robots[r_id].get_end_effector_pose()

        # Object related information
        object_states = {}
        all_object_ids = set()
        if orbit:
            all_object_ids.update(orbit.grasp_poses.keys())
            all_object_ids.update(orbit.placement_poses.keys())
        for object_id in all_object_ids:
            is_attached = object_id in orbit.attachments.values()
            if object_id in configuration.placement_poses:
                obj_status = 'placed'
                obj_pose = configuration.placement_poses[object_id]
            elif object_id in configuration.grasp_poses:
                obj_status = 'grasped'
                obj_pose = configuration.grasp_poses[object_id]
            else:
                obj_status = 'grasped' if is_attached else 'placed'
                obj_pose = orbit.grasp_poses.get(object_id)
                if obj_pose is None:
                    obj_pose = orbit.placement_poses.get(object_id)
                if obj_pose is None:
                    raise ValueError(f"Object{object_id} has no defined pose in the orbit, which is wrong")
            object_states[object_id] = {
                'status': obj_status,
                'pose': obj_pose,
            }

        attachment_sim_ids = {}
        placed_objects_sim_ids = []
        # set object poses in the simulator
        for object_id, state in object_states.items():
            if state['status'] == 'grasped':
                holding_robot_id = None
                for r_id, holding_object in configuration.attachments.items():
                    if holding_object == object_id:
                        holding_robot_id = r_id
                        break
                if holding_robot_id is not None:
                    t_object = t_ee_poses[holding_robot_id] @ state['pose'] 
                    self.mobjects[object_id].set_pose(t_object)
                    attachment_sim_ids[self.robots[holding_robot_id].sim_id] = self.mobjects[object_id].sim_id
                # when grasping, at the boundary point the object still touching the table
                placed_objects_sim_ids.append(self.mobjects[object_id].sim_id)
            else:
                self.mobjects[object_id].set_pose(state['pose'])
                placed_objects_sim_ids.append(self.mobjects[object_id].sim_id)
                # when placing, at the boundary point the object still touch the gripper
                robots_holding_object = [r_id for r_id, holding_object in orbit.attachments.items() if holding_object == object_id]
                for r_id in robots_holding_object:
                    attachment_sim_ids[self.robots[r_id].sim_id] = self.mobjects[object_id].sim_id


        all_robot_sim_ids = [r.sim_id for r in self.robots]
        if self._has_forbidden_collision(
            all_robot_sim_ids,
            attachment_sim_ids,
            placed_objects_sim_ids,
            debug_context="boundary",
        ):
            return False
        return True


    def is_pre_pose_feasible(self, robot_id, pre_pose_joint_angles, transition_joint_angles, orbit):
        """
        Check if the given pre-pose configuration is feasible for the robot before performing a transition action.
        Check collision of prepose and motion between prepose and transition pose.
        
        :param robot_id: ID of the robot
        :param pre_pose_joint_angles: Joint angles for the pre-pose configuration
        :param transition_joint_angles: Joint angles for the transition configuration
        :param orbit: The orbit context to check feasibility against

        :return: True if feasible, False otherwise
        """
        
        robot_sim_id = self.robots[robot_id].sim_id

        # set mobjects pose
        for obj_id, placement_pose in orbit.placement_poses.items():
            self.mobjects[obj_id].set_pose(
                placement_pose
            )
        
        # function to check feasibility of each specific joint_angles
        def is_feasible_joint_angles(joint_angles):
            # Check if the configuration is feasible in the context of the orbit
            if self.is_self_colliding(robot_id, joint_angles):
                return False
            # Set gripper state based on attachment
            gripper_target = 0.0 if orbit.attachments.get(robot_id) is not None else 0.04
            p.resetJointState(self.robots[robot_id].sim_id, 9, gripper_target, physicsClientId=self.using_client)
            p.resetJointState(self.robots[robot_id].sim_id, 10, gripper_target, physicsClientId=self.using_client)
            # check if collision with the environment
            base_links = set(self.robots[robot_id].base_link_index)
            for fixed_sim_id in self.fixed_objects:
                contacts = p.getClosestPoints(
                    robot_sim_id,
                    fixed_sim_id,
                    distance=0.0,
                    physicsClientId=self.using_client,
                )
                for c in contacts:
                    if c[8] >= SAFE_APPROACH_DISTANCE:
                        continue
                    robot_link = c[3] if c[1] == robot_sim_id else c[4]
                    if robot_link in base_links:
                        continue
                    if robot_link in self.robots[robot_id].fingertip_link_indices and c[8] >= SAFE_APPROACH_DISTANCE/10: # allow close contact with fingertip links
                        continue
                    if print_debug:
                        print(f"Grasp pose for Robot{robot_id} has forbidden collision with environment object {fixed_sim_id}.")
                        print(f"Contact details: {c}")
                    return False
            return True
            

        # check pre_pose 
        if not is_feasible_joint_angles(pre_pose_joint_angles):
            return False
        # TODO: check motion between pre_pose and transition pose. Here we just offset 10cm so now we assume it's collision free

        return True
    
    

    def collision_free_motion(self, config1, config2, orbit=None, resolution=None, num_checks=None) -> bool:
        """
            config1, config2 can be either:
            - Configuration objects
            - joint angle lists / numpy arrays

            :param orbit: the orbit context to check feasibility against
            :param resolution: the maximum allowed distance between neighboring joint samples for collision checking
            :param num_checks: legacy parameter for backward compatibility, specifying the number of checks along the straight-line motion. 

            :return: True if the motion is collision-free, False otherwise
        """

        if resolution is None:
            resolution = self.collision_checking_resolution

        if resolution <= 0:
            raise ValueError("resolution must be > 0")

        # Extract joint angles
        q1 = config1.joint_angles if hasattr(config1, "joint_angles") else config1
        q2 = config2.joint_angles if hasattr(config2, "joint_angles") else config2

        q_start = np.ravel(q1)
        q_goal = np.ravel(q2)

        total_motion = np.linalg.norm(q_goal - q_start)

        intervals = [(0.0, 1.0, q1, q2)]

        while intervals:
            t_left, t_right, q_left, q_right = intervals.pop()

            if np.max(np.abs(np.array(q_right) - np.array(q_left))) < self.collision_joint_resolution:
                continue

            step_distance = np.linalg.norm(np.ravel(q_right) - np.ravel(q_left))
            if step_distance <= resolution:
                continue

            t_mid = 0.5 * (t_left + t_right)
            q_step = interpolate_motion(q1, q2, t_mid)

            if not self.is_feasible_in_orbit(q_step, orbit):
                return False

            intervals.append((t_left, t_mid, q_left, q_step))
            intervals.append((t_mid, t_right, q_step, q_right))

        return True
    

    def constituent_motion_collision_checking(self, robot_id, node_start, node_goal, orbit=None, resolution=None) -> bool:
        """
        Check if the motion between two configurations for a specific robot is collision-free.

        Level 1 collision checking in dRRT* to check:
            - Robot self-collision
            - Robot-environment collision (including fixed objects and plane)
            - Robot carrying object collision with environment

        :param robot_id: ID of the robot to check
        :param q_start: Starting joint angles for the robot
        :param q_goal: Goal joint angles for the robot
        :param orbit: The orbit context to check feasibility against
        :param resolution: The maximum allowed distance between neighboring joint samples for collision checking
        :return: True if the motion is collision-free, False otherwise
        """

        q_start = (
            node_start.pre_pose
            if node_start.is_transition and node_start.pre_pose is not None
            else node_start.joint_angles
        )
        q_goal = (
            node_goal.pre_pose
            if node_goal.is_transition and node_goal.pre_pose is not None
            else node_goal.joint_angles
        )

        if resolution is None:
            resolution = self.collision_checking_resolution

        if resolution <= 0:
            raise ValueError("resolution must be > 0")

        # MAYBE this two already checked
        # if not self.constituent_node_collision_checking(robot_id, q_start, orbit):
        #     return False
        # if not self.constituent_node_collision_checking(robot_id, q_goal, orbit):
        #     return False

        intervals = [(0.0, 1.0, q_start, q_goal)]

        while intervals:
            t_left, t_right, q_left, q_right = intervals.pop()

            if np.max(np.abs(np.asarray(q_right) - np.asarray(q_left))) < self.collision_joint_resolution:
                continue

            if np.linalg.norm(np.ravel(q_right) - np.ravel(q_left)) <= resolution:
                continue

            t_mid = 0.5 * (t_left + t_right)
            q_mid = interpolate_motion(q_start, q_goal, t_mid)

            if not self.constituent_node_collision_checking(robot_id, q_mid, orbit):
                return False

            intervals.append((t_left, t_mid, q_left, q_mid))
            intervals.append((t_mid, t_right, q_mid, q_right))

        return True


    def _ensure_collision_acceleration_state(self):
        """Initialize shared caches and invalidate them after a scene reset.

        Several derived environments build their PyBullet worlds without
        calling ``SimplePickPlaceEnv.__init__``.  Initialize these fields here
        so every subclass using the shared collision implementation gets the
        same acceleration state safely.
        """
        if not hasattr(self, "_collision_world_signature"):
            self._collision_world_signature = None
            self._constituent_collision_contexts = {}
            self._tensor_collision_contexts = {}
            self._constituent_node_validity_cache = {}
            self._tensor_node_validity_cache = {}
            self._fixed_body_aabbs = {}

        signature = (
            self.using_client,
            tuple(id(robot) for robot in self.robots),
            tuple(id(obj) for obj in self.mobjects),
            tuple(getattr(self, "fixed_objects", ())),
            self.plane,
        )
        if signature == self._collision_world_signature:
            return

        self._collision_world_signature = signature
        self._constituent_collision_contexts.clear()
        self._tensor_collision_contexts.clear()
        self._constituent_node_validity_cache.clear()
        self._tensor_node_validity_cache.clear()
        self._fixed_body_aabbs = {
            body_id: self._whole_body_aabb(body_id)
            for body_id in getattr(self, "fixed_objects", ())
            if body_id is not None
        }

    def _whole_body_aabb(self, body_id):
        """Return an AABB covering a body's base and every child link."""
        link_indices = range(
            -1,
            p.getNumJoints(body_id, physicsClientId=self.using_client),
        )
        link_aabbs = [
            p.getAABB(
                body_id,
                linkIndex=link_index,
                physicsClientId=self.using_client,
            )
            for link_index in link_indices
        ]
        return (
            tuple(min(aabb[0][axis] for aabb in link_aabbs) for axis in range(3)),
            tuple(max(aabb[1][axis] for aabb in link_aabbs) for axis in range(3)),
        )

    @staticmethod
    def _aabbs_overlap(first, second, margin=0.0):
        """Return whether two axis-aligned boxes overlap within ``margin``."""
        lower_a, upper_a = first
        lower_b, upper_b = second
        return all(
            upper_a[axis] + margin >= lower_b[axis]
            and upper_b[axis] + margin >= lower_a[axis]
            for axis in range(3)
        )

    @staticmethod
    def _configuration_cache_key(configuration, decimals=8):
        values = np.asarray(configuration, dtype=float).reshape(-1)
        return tuple(np.round(values, decimals=decimals))

    def _constituent_collision_context(self, robot_id, orbit):
        self._ensure_collision_acceleration_state()
        key = (robot_id, id(orbit) if orbit is not None else -1)
        cached = self._constituent_collision_contexts.get(key)
        if cached is not None:
            return cached

        robot = self.robots[robot_id]
        carried_object_id = (
            orbit.attachments.get(robot_id) if orbit is not None else None
        )
        placed_object_poses = {
            object_id: placement_pose
            for object_id, placement_pose in self._placed_object_poses_for_collision(
                orbit
            ).items()
            if object_id != carried_object_id
        }
        context = {
            "robot": robot,
            "robot_sim_id": robot.sim_id,
            "base_links": frozenset(robot.base_link_index),
            "static_bodies": tuple(
                body_id
                for body_id in (*getattr(self, "fixed_objects", ()), self.plane)
                if body_id is not None
            ),
            "carried_object_id": carried_object_id,
            "carried_object": (
                self.mobjects[carried_object_id]
                if carried_object_id is not None
                else None
            ),
            "grasp_pose": (
                orbit.grasp_poses.get(carried_object_id)
                if carried_object_id is not None
                else None
            ),
            "gripper_target": 0.0 if carried_object_id is not None else 0.04,
            "placed_object_poses": placed_object_poses,
        }
        self._constituent_collision_contexts[key] = context
        return context

    def _apply_constituent_scene(self, context, q_sample):
        for object_id, placement_pose in context["placed_object_poses"].items():
            self.mobjects[object_id].set_pose(placement_pose)

        robot = context["robot"]
        robot.set_pose(q_sample)
        for gripper_joint_id in (9, 10):
            p.resetJointState(
                context["robot_sim_id"],
                gripper_joint_id,
                context["gripper_target"],
                physicsClientId=self.using_client,
            )

        carried_object = context["carried_object"]
        grasp_pose = context["grasp_pose"]
        if carried_object is not None and grasp_pose is not None:
            carried_object.set_pose(robot.get_end_effector_pose() @ grasp_pose)

    def constituent_node_collision_checking(self, robot_id, q_sample, orbit=None) -> bool:
        """
        Check if a single configuration for a specific robot is collision-free.

        Level 1 collision checking in dRRT* to check:
            - Robot self-collision
            - Robot-environment collision (including fixed objects and plane)
            - Robot carrying object collision with environment

        :param robot_id: ID of the robot to check
        :param q_sample: Joint angles for the robot
        :param orbit: The orbit context to check feasibility against

        :return: True if the configuration is collision-free, False otherwise
        """

        context = self._constituent_collision_context(robot_id, orbit)
        cache_key = (
            robot_id,
            id(orbit) if orbit is not None else -1,
            self._configuration_cache_key(q_sample),
        )
        cached = self._constituent_node_validity_cache.get(cache_key)
        if cached is not None:
            # Preserve the scene-setting side effect expected by subsequent
            # collision checks while avoiding all pair queries.
            self._apply_constituent_scene(context, q_sample)
            return cached

        # Set robot to the test configuration, self collision checking.
        if self.is_self_colliding(robot_id, q_sample):
            if print_debug:
                print(f"Configuration for Robot{robot_id} is in self-collision.")
            self._constituent_node_validity_cache[cache_key] = False
            return False
        
        # Robot vs environment collision checking
        robot = context["robot"]
        robot_sim_id = context["robot_sim_id"]
        base_links = context["base_links"]

        # is_self_colliding() sets the arm joints. Set the gripper as well so
        # collision results do not depend on a previous simulation query.
        carrying_object_id = context["carried_object_id"]
        gripper_target = context["gripper_target"]
        for gripper_joint_id in (9, 10):
            p.resetJointState(
                robot_sim_id,
                gripper_joint_id,
                gripper_target,
                physicsClientId=self.using_client,
            )

        # getClosestPoints performs a pair-specific collision query.  This is
        # cheaper here than updating the contact cache for the entire world
        # with performCollisionDetection(), since only one robot and the
        # static environment bodies are relevant.
        static_bodies = context["static_bodies"]
        for environment_body_id in static_bodies:
            if environment_body_id is None:
                continue

            contacts = p.getClosestPoints(
                bodyA=robot_sim_id,
                bodyB=environment_body_id,
                distance=0.0,
                physicsClientId=self.using_client,
            )
            for contact in contacts:
                robot_link = contact[3]  # bodyA is always the robot

                # Preserve the intended robot/table support contact.  Plane
                # contact is not exempted.
                if (
                    environment_body_id in self.fixed_objects
                    and robot_link in base_links
                ):
                    continue

                if print_debug:
                    print(
                        f"Robot{robot_id} link {robot_link} is too close to "
                        f"static body {environment_body_id}; "
                        f"distance={contact[8]:.6f}."
                    )
                self._constituent_node_validity_cache[cache_key] = False
                return False

        # Placed movable objects are static obstacles within this orbit.
        for object_id, placement_pose in context["placed_object_poses"].items():
            placed_object = self.mobjects[object_id]
            if self._broad_phase_separated(
                robot.base_pos,
                robot.max_reach,
                np.asarray(placement_pose, dtype=float)[:3, 3],
                self._object_bounding_radius(placed_object),
            ):
                continue
            contacts = p.getClosestPoints(
                bodyA=robot_sim_id,
                bodyB=placed_object.sim_id,
                distance=0.0,
                physicsClientId=self.using_client,
            )
            if contacts:
                if print_debug:
                    print(
                        f"Robot{robot_id} and placed Object{object_id} are in "
                        f"collision; distance={contacts[0][8]:.6f}."
                    )
                self._constituent_node_validity_cache[cache_key] = False
                return False

        # Robot carrying object vs environment collision checking
        if carrying_object_id is not None:
            grasp_pose = context["grasp_pose"]
            if grasp_pose is None:
                if print_debug:
                    print(
                        f"Orbit has no grasp pose for Object{carrying_object_id} "
                        f"carried by Robot{robot_id}."
                    )
                self._constituent_node_validity_cache[cache_key] = False
                return False

            # Synchronize the carried object with q_sample before querying it.
            # Otherwise PyBullet would use the pose left by a previous check.
            t_ee = robot.get_end_effector_pose()
            t_object = t_ee @ grasp_pose
            context["carried_object"].set_pose(t_object)

            carrying_object_sim_id = context["carried_object"].sim_id
            carrying_object_aabb = p.getAABB(
                carrying_object_sim_id,
                physicsClientId=self.using_client,
            )
            for environment_body_id in static_bodies:
                if environment_body_id is None:
                    continue

                fixed_aabb = self._fixed_body_aabbs.get(environment_body_id)
                if (
                    fixed_aabb is not None
                    and not self._aabbs_overlap(
                        carrying_object_aabb,
                        fixed_aabb,
                    )
                ):
                    continue

                contacts = p.getClosestPoints(
                    bodyA=carrying_object_sim_id,
                    bodyB=environment_body_id,
                    distance=0.0,
                    physicsClientId=self.using_client,
                )
                for contact in contacts:
                    if print_debug:
                        print(
                            f"Robot{robot_id} carrying Object{carrying_object_id} is too close to "
                            f"static body {environment_body_id}; "
                            f"distance={contact[8]:.6f}."
                        )
                    self._constituent_node_validity_cache[cache_key] = False
                    return False


            # A carried object may not pass through an already placed object.
            for object_id, placement_pose in context["placed_object_poses"].items():
                placed_object = self.mobjects[object_id]
                placed_position = np.asarray(
                    placement_pose,
                    dtype=float,
                )[:3, 3]
                if self._broad_phase_separated(
                    np.asarray(t_object, dtype=float)[:3, 3],
                    self._object_bounding_radius(context["carried_object"]),
                    placed_position,
                    self._object_bounding_radius(placed_object),
                ):
                    continue
                contacts = p.getClosestPoints(
                    bodyA=carrying_object_sim_id,
                    bodyB=placed_object.sim_id,
                    distance=0.0,
                    physicsClientId=self.using_client,
                )
                if contacts:
                    if print_debug:
                        print(
                            f"Carried Object{carrying_object_id} and placed "
                            f"Object{object_id} are in collision; "
                            f"distance={contacts[0][8]:.6f}."
                        )
                    self._constituent_node_validity_cache[cache_key] = False
                    return False

        self._constituent_node_validity_cache[cache_key] = True
        return True
        

    def tensor_product_motion_collision_checking(self, v_start, v_goal, orbit=None, resolution=None) -> bool:

        """
        Check if the motion between two configurations for a specific robot is collision-free.

        Level 2 collision checking in dRRT* to check:
            - Robot vs robot collision
            - Robot vs carrying objects of other robots
            - Moving objects vs moving objects collision

        :param v_start: Starting vertex containing all robot's joint angles
        :param v_goal: Goal vertex containing all robot's joint angles
        :param orbit: The orbit context to check feasibility against
        :param resolution: The maximum allowed distance between neighboring joint samples for collision checking
        :return: True if the motion is collision-free, False otherwise
        """

        q_start = v_start.joint_angles
        q_goal = v_goal.joint_angles

        expected_shape = (len(self.robots), 7)
        if np.asarray(q_start).shape != expected_shape or np.asarray(q_goal).shape != expected_shape:
            raise ValueError(
                f"tensor-product configurations must have shape {expected_shape}"
            )

        if resolution is None:
            resolution = self.collision_checking_resolution

        if resolution <= 0:
            raise ValueError("resolution must be > 0")

        if not self.tensor_product_node_collision_checking(q_start, orbit):
            return False
        if not self.tensor_product_node_collision_checking(q_goal, orbit):
            return False

        intervals = [(0.0, 1.0, q_start, q_goal)]

        while intervals:
            t_left, t_right, q_left, q_right = intervals.pop()

            if np.max(np.abs(np.asarray(q_right) - np.asarray(q_left))) < self.collision_joint_resolution:
                continue

            if np.linalg.norm(np.ravel(q_right) - np.ravel(q_left)) <= resolution:
                continue

            t_mid = 0.5 * (t_left + t_right)
            q_mid = interpolate_motion(q_start, q_goal, t_mid)

            if not self.tensor_product_node_collision_checking(composite_joint_angles=q_mid, orbit=orbit):
                return False

            intervals.append((t_left, t_mid, q_left, q_mid))
            intervals.append((t_mid, t_right, q_mid, q_right))

        return True


    def _tensor_collision_context(self, orbit):
        self._ensure_collision_acceleration_state()
        key = id(orbit) if orbit is not None else -1
        cached = self._tensor_collision_contexts.get(key)
        if cached is not None:
            return cached

        attachments = dict(orbit.attachments) if orbit is not None else {}
        carried_by_robot = {
            robot_id: object_id
            for robot_id, object_id in attachments.items()
            if object_id is not None
        }
        carried_object_ids = tuple(sorted(set(carried_by_robot.values())))
        robot_pairs = tuple(
            (self.robots[first], self.robots[second])
            for first in range(len(self.robots))
            for second in range(first + 1, len(self.robots))
            if not self._broad_phase_separated(
                self.robots[first].base_pos,
                self.robots[first].max_reach,
                self.robots[second].base_pos,
                self.robots[second].max_reach,
            )
        )
        robot_object_pairs = tuple(
            (robot, self.mobjects[object_id])
            for robot_id, robot in enumerate(self.robots)
            for object_id in carried_object_ids
            if carried_by_robot.get(robot_id) != object_id
        )
        object_pairs = tuple(
            (self.mobjects[carried_object_ids[first]],
             self.mobjects[carried_object_ids[second]])
            for first in range(len(carried_object_ids))
            for second in range(first + 1, len(carried_object_ids))
        )
        context = {
            "attachments": attachments,
            "carried_by_robot": carried_by_robot,
            "grasp_poses": (
                dict(orbit.grasp_poses) if orbit is not None else {}
            ),
            "robot_pairs": robot_pairs,
            "robot_object_pairs": robot_object_pairs,
            "object_pairs": object_pairs,
            "object_positions": {},
        }
        self._tensor_collision_contexts[key] = context
        return context

    @staticmethod
    def _broad_phase_separated(center_a, radius_a, center_b, radius_b):
        """Return True only when conservative bounding spheres are disjoint."""
        return (
            np.linalg.norm(
                np.asarray(center_a, dtype=float)
                - np.asarray(center_b, dtype=float)
            )
            > float(radius_a) + float(radius_b)
        )

    def _object_bounding_radius(self, moving_object):
        radius = getattr(moving_object, "bounding_radius", None)
        if radius is not None:
            return float(radius)
        size = getattr(moving_object, "size", None)
        if size is not None:
            return 0.5 * np.sqrt(3.0) * float(size)
        aabb_min, aabb_max = p.getAABB(
            moving_object.sim_id,
            physicsClientId=self.using_client,
        )
        return 0.5 * np.linalg.norm(
            np.asarray(aabb_max, dtype=float)
            - np.asarray(aabb_min, dtype=float)
        )

    def _apply_tensor_scene(self, context, composite_joint_angles):
        for robot_id, q_sample in enumerate(composite_joint_angles):
            robot = self.robots[robot_id]
            robot.set_pose(q_sample)

            holding_object = context["carried_by_robot"].get(robot_id)
            gripper_target = 0.0 if holding_object is not None else 0.04
            for gripper_joint_id in (9, 10):
                p.resetJointState(
                    robot.sim_id,
                    gripper_joint_id,
                    gripper_target,
                    physicsClientId=self.using_client,
                )

            if holding_object is None:
                continue
            grasp_pose = context["grasp_poses"].get(holding_object)
            if grasp_pose is None:
                return False
            object_pose = robot.get_end_effector_pose() @ grasp_pose
            self.mobjects[holding_object].set_pose(object_pose)
            context["object_positions"][holding_object] = (
                np.asarray(object_pose, dtype=float)[:3, 3].copy()
            )
        return True

    def tensor_product_node_collision_checking(self, composite_joint_angles, orbit=None) -> bool:
        """
        Check if a single tensor-product configuration for all robots is collision-free.

        Level 2 collision checking in dRRT* to check:
            - Robot vs robot collision
            - Robot vs carrying objects of other robots
            - Moving objects vs moving objects collision
        """

        expected_shape = (len(self.robots), 7)
        if np.asarray(composite_joint_angles).shape != expected_shape:
            raise ValueError(
                f"composite_joint_angles must have shape {expected_shape}"
            )

        context = self._tensor_collision_context(orbit)
        cache_key = (
            id(orbit) if orbit is not None else -1,
            self._configuration_cache_key(composite_joint_angles),
        )
        cached = self._tensor_node_validity_cache.get(cache_key)
        if cached is not None:
            self._apply_tensor_scene(context, composite_joint_angles)
            return cached

        if not self._apply_tensor_scene(context, composite_joint_angles):
            if print_debug:
                print("Orbit is missing a grasp pose for a carried object.")
            self._tensor_node_validity_cache[cache_key] = False
            return False


        # check pairwise robot-robot collision
        for robot_i, robot_j in context["robot_pairs"]:
            contacts = p.getClosestPoints(
                bodyA=robot_i.sim_id,
                bodyB=robot_j.sim_id,
                distance=0.0,
                physicsClientId=self.using_client,
            )
            if contacts:
                if print_debug:
                    print(
                        f"Robot{robot_i.id} and Robot{robot_j.id} are in "
                        f"collision; distance={contacts[0][8]:.6f}."
                    )
                self._tensor_node_validity_cache[cache_key] = False
                return False


        # check pairwise robot-object collision
        for robot, moving_object in context["robot_object_pairs"]:
            if self._broad_phase_separated(
                robot.base_pos,
                robot.max_reach,
                context["object_positions"][moving_object.id],
                self._object_bounding_radius(moving_object),
            ):
                continue
            contacts = p.getClosestPoints(
                bodyA=robot.sim_id,
                bodyB=moving_object.sim_id,
                distance=0.0,
                physicsClientId=self.using_client,
            )
            if contacts:
                if print_debug:
                    print(
                        f"Robot{robot.id} and Object{moving_object.id} are in "
                        f"collision; distance={contacts[0][8]:.6f}."
                    )
                self._tensor_node_validity_cache[cache_key] = False
                return False


        # check pairwise object-object collision
        object_aabbs = {}
        for object_i, object_j in context["object_pairs"]:
            if self._broad_phase_separated(
                context["object_positions"][object_i.id],
                self._object_bounding_radius(object_i),
                context["object_positions"][object_j.id],
                self._object_bounding_radius(object_j),
            ):
                continue
            if object_i.sim_id not in object_aabbs:
                object_aabbs[object_i.sim_id] = p.getAABB(
                    object_i.sim_id,
                    physicsClientId=self.using_client,
                )
            if object_j.sim_id not in object_aabbs:
                object_aabbs[object_j.sim_id] = p.getAABB(
                    object_j.sim_id,
                    physicsClientId=self.using_client,
                )
            aabb_i = object_aabbs[object_i.sim_id]
            aabb_j = object_aabbs[object_j.sim_id]
            if not self._aabbs_overlap(aabb_i, aabb_j):
                continue
            contacts = p.getClosestPoints(
                bodyA=object_i.sim_id,
                bodyB=object_j.sim_id,
                distance=0.0,
                physicsClientId=self.using_client,
            )
            if contacts:
                if print_debug:
                    print(
                        f"Object{object_i.id} and Object{object_j.id} are in "
                        f"collision; distance={contacts[0][8]:.6f}."
                    )
                self._tensor_node_validity_cache[cache_key] = False
                return False

        self._tensor_node_validity_cache[cache_key] = True
        return True





    ###################################################################
    #======================= Helper function for computation =======================
    ####################################################################
    def _profile_start(self):
        return time.perf_counter()

    def _profile_end(self, key, start_time):
        self.profile_time[key] += time.perf_counter() - start_time
        self.profile_calls[key] += 1

    def print_profile_summary(self):
        print("\n=== Profiling Summary ===")
        for key in sorted(self.profile_time, key=self.profile_time.get, reverse=True):
            total = self.profile_time[key]
            calls = self.profile_calls[key]
            avg = total / calls if calls else 0.0
            print(f"{key}: total={total:.4f}s calls={calls} avg={avg:.6f}s")


    ####################################################################
    #======================= Visualization Utils =======================
    ####################################################################
    def _draw_object_targets(self, client_id=None):
        """Visualize target object locations"""
        if client_id is None:
            client_id = p.connect(p.GUI)
        for obj in self.mobjects:
            color = cube_colors[obj.id % len(cube_colors)][:3]

            p.addUserDebugPoints(
                pointPositions=[[obj.x_goal, obj.y_goal, obj.z_goal]],
                pointColorsRGB=[color],
                pointSize=12,
                physicsClientId=client_id
            )

    # For debugging: visualize robot frames
    def draw_frame(self, pos, orn, length=0.1, life_time=0):
        length=0.5
        line_width=5

        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        origin = np.array(pos)

        x_axis = origin + rot[:, 0] * length
        y_axis = origin + rot[:, 1] * length
        z_axis = origin + rot[:, 2] * length

        # X = red
        p.addUserDebugLine(origin, x_axis, [1, 0, 0], line_width, lifeTime=life_time, physicsClientId=self.using_client)
        # Y = green
        p.addUserDebugLine(origin, y_axis, [0, 1, 0], line_width, lifeTime=life_time, physicsClientId=self.using_client)
        # Z = blue
        p.addUserDebugLine(origin, z_axis, [0, 0, 1], line_width, lifeTime=life_time, physicsClientId=self.using_client)


    def _open_final_path_gui(self, width=1920, height=1080):
        """Open a fresh, large GUI client for final-path playback."""
        if self.using_client is not None and p.isConnected(self.using_client):
            p.disconnect(self.using_client)

        client_id = p.connect(
            p.GUI,
            options=f"--width={width} --height={height}",
        )
        if client_id < 0:
            raise RuntimeError("Failed to open the PyBullet visualization GUI.")

        self.using_client = client_id
        return client_id



    def simulate_final_path(
        self,
        output_path_filename,
        output_sim_filename="simulation.mp4",
        delay=0.2,
        num_interpolate_points=10,
        transition_pause=0.5,
    ):
        """Load and replay an orbit-structured path while recording an MP4.

        ``transition_pause`` holds a grasp/place endpoint after its attachment
        update.  Composite robot motion remains parallel, but distinct symbolic
        transitions no longer appear to occur in the same instant.
        """
        def _parse_matrix(block: str):
            nums = np.fromstring(block.replace('[', ' ').replace(']', ' '), sep=' ')
            if nums.size != 16:
                raise ValueError(f"Expected 16 values in matrix, got {nums.size}")
            return nums.reshape(4, 4)

        def _parse_configuration_section(sec: str):
            m = re.search(r"(?m)^\s*Joint Angles: (.+)$", sec)
            if not m:
                return None
            joint_angles = ast.literal_eval(m.group(1).strip())

            m = re.search(r"(?m)^\s*Attachments: (.+)$", sec)
            attachments = ast.literal_eval(m.group(1).strip()) if m else {}

            m = re.search(r"(?m)^\s*Is Transition: (True|False)$", sec)
            is_transition = m.group(1) == "True" if m else False

            grasp_poses = {}
            placement_poses = {}

            # Old format: "Object X grasp pose:" / "Object X placement pose:"
            for obj_id, block in re.findall(r"Object (\d+) grasp pose:\n((?:\s*\[[^\n]*\n){4})", sec):
                grasp_poses[int(obj_id)] = _parse_matrix(block)
            for obj_id, block in re.findall(r"Object (\d+) placement pose:\n((?:\s*\[[^\n]*\n){4})", sec):
                placement_poses[int(obj_id)] = _parse_matrix(block)

            # New format per-configuration details: "Object X:" with matrix
            grasp_block_match = re.search(r"(?s)\bGrasp Poses:[^\n]*\n(.*?)(?:\n\s*Placement Poses:|\n\s*q\d+ \(Configuration|\Z)", sec)
            if grasp_block_match:
                grasp_block = grasp_block_match.group(1)
                for obj_id, block in re.findall(r"Object\s+(\d+):\n((?:\s*\[[^\n]*\n){4})", grasp_block):
                    grasp_poses[int(obj_id)] = _parse_matrix(block)

            place_block_match = re.search(r"(?s)\bPlacement Poses:[^\n]*\n(.*?)(?:\n\s*q\d+ \(Configuration|\Z)", sec)
            if place_block_match:
                place_block = place_block_match.group(1)
                for obj_id, block in re.findall(r"Object\s+(\d+):\n((?:\s*\[[^\n]*\n){4})", place_block):
                    placement_poses[int(obj_id)] = _parse_matrix(block)

            return Configuration(
                joint_angles=joint_angles,
                attachments=attachments,
                is_transition=is_transition,
                parent_orbit=None,
                grasp_poses=grasp_poses if is_transition else None,
                placement_poses=placement_poses if is_transition else None,
            )

        def _load_path_from_file(path_file: str):
            with open(path_file, 'r') as f:
                text = f.read()

            def _parse_orbit_attachments(raw_attachments):
                parsed = {}
                for r_id, value in raw_attachments.items():
                    robot_id = int(r_id)
                    if value is None:
                        parsed[robot_id] = None
                        continue
                    if isinstance(value, str):
                        if value.strip().lower() in {"none", "null"}:
                            parsed[robot_id] = None
                            continue
                        m = re.search(r"object\s+(\d+)", value)
                        if m:
                            parsed[robot_id] = int(m.group(1))
                            continue
                    parsed[robot_id] = value
                return parsed

            def _parse_orbit_metadata(block: str):
                metadata = {
                    "orbit_id": None,
                    "orbit_hex": None,
                    "robot_states": {},
                    "object_states": {},
                    "attachments": {},
                    "grasp_poses": {},
                    "placement_poses": {},
                }

                m = re.search(r"(?m)^Orbit Information: id=(\d+),\s*hex=(0x[0-9a-fA-F]+)", block)
                if m:
                    metadata["orbit_id"] = int(m.group(1))
                    metadata["orbit_hex"] = m.group(2)

                m = re.search(r"(?m)^\s*Robot States:\s*(\{.*\})", block)
                if m:
                    try:
                        raw_robot_states = ast.literal_eval(m.group(1).strip())
                        metadata["robot_states"] = {int(k): v for k, v in raw_robot_states.items()}
                    except (SyntaxError, ValueError):
                        metadata["robot_states"] = {}

                m = re.search(r"(?m)^\s*Object States:\s*(\{.*\})", block)
                if m:
                    try:
                        raw_object_states = ast.literal_eval(m.group(1).strip())
                        metadata["object_states"] = {int(k): v for k, v in raw_object_states.items()}
                    except (SyntaxError, ValueError):
                        metadata["object_states"] = {}

                m = re.search(r"(?m)^\s*Attachments:\s*(\{.*\})", block)
                if m:
                    try:
                        raw_attachments = ast.literal_eval(m.group(1).strip())
                        metadata["attachments"] = _parse_orbit_attachments(raw_attachments)
                    except (SyntaxError, ValueError):
                        metadata["attachments"] = {}

                grasp_section = re.search(
                    r"(?ms)^\s*Grasp Pose Matrices:\n(.*?)(?=^\s*Placement Pose Matrices:|^\s*Configurations in this orbit path:|\Z)",
                    block,
                )
                if grasp_section:
                    for obj_id, matrix_block in re.findall(
                        r"(?ms)^\s*Object\s+(\d+):\n(\s*\[\[.*?\]\])",
                        grasp_section.group(1),
                    ):
                        metadata["grasp_poses"][int(obj_id)] = _parse_matrix(matrix_block)

                place_section = re.search(
                    r"(?ms)^\s*Placement Pose Matrices:\n(.*?)(?=^\s*Configurations in this orbit path:|\Z)",
                    block,
                )
                if place_section:
                    for obj_id, matrix_block in re.findall(
                        r"(?ms)^\s*Object\s+(\d+):\n(\s*\[\[.*?\]\])",
                        place_section.group(1),
                    ):
                        metadata["placement_poses"][int(obj_id)] = _parse_matrix(matrix_block)

                return metadata

            orbit_path = []

            # New orbit-structured format
            orbit_blocks = re.findall(r"(?ms)^Orbit\d+:.*?(?=^Orbit\d+:|\Z)", text)
            if orbit_blocks:
                for block in orbit_blocks:
                    orbit_header_match = re.search(r"(?m)^(Orbit\d+):", block)
                    orbit_label = orbit_header_match.group(1) if orbit_header_match else "Orbit"
                    segment_configs = []

                    config_matches = re.finditer(
                        r"(?ms)^\s*q\d+ \(Configuration \d+\):\n(.*?)(?=^\s*q\d+ \(Configuration \d+\):|\Z)",
                        block,
                    )
                    for m in config_matches:
                        cfg = _parse_configuration_section(m.group(1))
                        if cfg is not None:
                            segment_configs.append(cfg)

                    if segment_configs:
                        orbit_path.append(
                            {
                                "orbit": orbit_label,
                                "configs": segment_configs,
                                **_parse_orbit_metadata(block),
                            }
                        )
            # Backward-compatible flat format fallback
            if not orbit_path:
                sections = re.split(r"(?m)^--- Configuration \d+ ---\n", text)
                flat_configs = []
                for sec in sections[1:]:
                    cfg = _parse_configuration_section(sec)
                    if cfg is not None:
                        flat_configs.append(cfg)
                if flat_configs:
                    flat_attachments = dict(flat_configs[0].attachments) if flat_configs[0].attachments else {}
                    flat_grasp_poses = {}
                    flat_placement_poses = {}
                    for cfg in flat_configs:
                        if cfg.grasp_poses:
                            flat_grasp_poses.update(cfg.grasp_poses)
                        if cfg.placement_poses:
                            flat_placement_poses.update(cfg.placement_poses)
                    orbit_path = [
                        {
                            "orbit": "Orbit1",
                            "configs": flat_configs,
                            "orbit_id": None,
                            "orbit_hex": None,
                            "robot_states": {},
                            "object_states": {},
                            "attachments": flat_attachments,
                            "grasp_poses": flat_grasp_poses,
                            "placement_poses": flat_placement_poses,
                        }
                    ]

            return orbit_path


        orbit_path = _load_path_from_file(str(output_path_filename))

        # print("Orbit path loaded from file:")
        # for segment in orbit_path:
        #     print(f"  {segment['orbit']}: {len(segment['configs'])} configurations")
        #     print(f"    orbit_id={segment.get('orbit_id')}, orbit_hex={segment.get('orbit_hex')}")
        #     print(f"    robot_states={segment.get('robot_states', {})}")
        #     print(f"    object_states={segment.get('object_states', {})}")
        #     print(f"    attachments={segment.get('attachments', {})}")
        #     print(f"    grasp_pose_objects={sorted(list(segment.get('grasp_poses', {}).keys()))}")
        #     print(f"    placement_pose_objects={sorted(list(segment.get('placement_poses', {}).keys()))}")
        #     print(f"    Sample config joint angles: {segment['configs'][0].joint_angles if segment['configs'] else 'N/A'}")

        total_configs = sum(len(segment.get("configs", [])) for segment in orbit_path)
        if total_configs < 2:
            print("Video not recorded: path must contain at least 2 configurations.")
            return None


        if str(output_sim_filename).lower().endswith(".mp4") and shutil.which("ffmpeg") is None:
            print("Video not recorded: ffmpeg is not installed. Install it and retry (Ubuntu: sudo apt install ffmpeg).")
            return None

        output_path = os.path.abspath(output_sim_filename)
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        self._open_final_path_gui(width=1920, height=1080)
        self.reset(client_id=self.using_client)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0.5, 0, 0.7],
            physicsClientId=self.using_client,
        )

        # ``reset`` leaves newly loaded robots at the URDF joint defaults,
        # which need not equal the first configuration in the saved path.
        # Initialize them before video logging; otherwise the initial snap to
        # q1 is recorded as spurious robot motion (especially noticeable for a
        # robot that is intentionally stationary in the first orbit).
        first_orbit = next(
            orbit for orbit in orbit_path if orbit.get("configs")
        )
        first_config = first_orbit["configs"][0]
        initial_attachments = dict(first_orbit.get("attachments") or {})
        initial_attachments.update(first_config.attachments or {})
        for r_id in range(self.num_robots):
            robot = self.robots[r_id]
            for joint_idx, angle in zip(
                robot.movable_joint_indices,
                first_config.joint_angles[r_id],
            ):
                p.resetJointState(
                    robot.sim_id,
                    joint_idx,
                    angle,
                    physicsClientId=self.using_client,
                )
            gripper_target = (
                0.0 if initial_attachments.get(r_id) is not None else 0.04
            )
            p.resetJointState(
                robot.sim_id,
                9,
                gripper_target,
                physicsClientId=self.using_client,
            )
            p.resetJointState(
                robot.sim_id,
                10,
                gripper_target,
                physicsClientId=self.using_client,
            )
        p.stepSimulation(physicsClientId=self.using_client)

        log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4,
            output_path,
            physicsClientId=self.using_client,
        )
        if log_id < 0:
            print(f"Failed to start video logging to {output_path}")
            return None


        try:
            current_pose = None
            current_scene_state = None

            def _resolve_scene_state(config, orbit_attachments, orbit_grasp_poses, orbit_placement_poses, use_config_overrides=True):
                attachments = dict(orbit_attachments)
                if use_config_overrides:
                    attachments.update(config.attachments or {})

                grasp_poses = dict(orbit_grasp_poses)
                placement_poses = dict(orbit_placement_poses)

                if use_config_overrides:
                    for obj_id, pose in (config.grasp_poses or {}).items():
                        grasp_poses[obj_id] = pose
                        placement_poses.pop(obj_id, None)
                    for obj_id, pose in (config.placement_poses or {}).items():
                        placement_poses[obj_id] = pose
                        grasp_poses.pop(obj_id, None)

                attached_object_ids = {obj_id for obj_id in attachments.values() if obj_id is not None}
                for obj_id in list(grasp_poses.keys()):
                    if obj_id not in attached_object_ids:
                        grasp_poses.pop(obj_id, None)
                for obj_id in attached_object_ids:
                    placement_poses.pop(obj_id, None)

                return {
                    "attachments": attachments,
                    "grasp_poses": grasp_poses,
                    "placement_poses": placement_poses,
                }

            def _apply_scene(q_sample, attachments, grasp_poses, placement_poses):
                # Move robots and set gripper opening based on whether each robot is holding an object.
                for r_id in range(self.num_robots):
                    robot_sim_id = self.robots[r_id].sim_id
                    for join_idx, angle in zip(self.robots[r_id].movable_joint_indices, q_sample[r_id]):
                        p.resetJointState(robot_sim_id, join_idx, angle, physicsClientId=self.using_client)
                    gripper_target = 0.0 if attachments.get(r_id) is not None else 0.04
                    p.resetJointState(robot_sim_id, 9, gripper_target, physicsClientId=self.using_client)
                    p.resetJointState(robot_sim_id, 10, gripper_target, physicsClientId=self.using_client)

                # Keep placed objects fixed by orbit-level placement poses.
                for obj_id, placement_pose in placement_poses.items():
                    self.mobjects[obj_id].set_pose(placement_pose)

                # Attach held objects to corresponding end-effectors.
                for robot_id, holding_object in attachments.items():
                    if holding_object is None:
                        continue
                    grasp_pose = grasp_poses.get(holding_object)
                    if grasp_pose is None:
                        continue
                    t_ee = self.robots[robot_id].get_end_effector_pose()
                    t_object = t_ee @ grasp_pose
                    self.mobjects[holding_object].set_pose(t_object)

            def _hold_transition(config, before_state, after_state):
                """Keep a changed attachment endpoint visible in the replay."""
                if not config.is_transition or transition_pause <= 0:
                    return
                if before_state["attachments"] == after_state["attachments"]:
                    return

                # One replay step already displays the endpoint. Add enough
                # frames to make the discrete grasp/place event unambiguous.
                frame_delay = delay if delay > 0 else 1.0 / 60.0
                hold_steps = max(1, int(np.ceil(transition_pause / frame_delay)))
                for _ in range(hold_steps):
                    _apply_scene(
                        config.joint_angles,
                        after_state["attachments"],
                        after_state["grasp_poses"],
                        after_state["placement_poses"],
                    )
                    p.stepSimulation(physicsClientId=self.using_client)
                    if delay > 0:
                        time.sleep(delay)

            for orbit in orbit_path:
                configs = orbit.get("configs", [])
                if not configs:
                    continue
                if current_pose is None:
                    current_pose = configs[0].joint_angles

                # Orbit-level source of truth for symbolic info.
                orbit_attachments = orbit.get("attachments") or (configs[0].attachments if configs[0].attachments else {})
                orbit_grasp_poses = orbit.get("grasp_poses", {})
                orbit_placement_poses = orbit.get("placement_poses", {})

                # Ensure all robots have explicit attachment entries.
                for r_id in range(self.num_robots):
                    orbit_attachments.setdefault(r_id, None)

                start_scene_state = _resolve_scene_state(
                    configs[0],
                    orbit_attachments,
                    orbit_grasp_poses,
                    orbit_placement_poses,
                )
                entry_scene_state = _resolve_scene_state(
                    configs[0],
                    orbit_attachments,
                    orbit_grasp_poses,
                    orbit_placement_poses,
                    use_config_overrides=False,
                )

                # Motion between the final configuration of previous orbit and the initial configuration of current orbit.
                if current_pose is not None:
                    q_start_orbit = configs[0].joint_angles
                    bridge_motion = np.linalg.norm(np.ravel(current_pose) - np.ravel(q_start_orbit))
                    if bridge_motion > 1e-8:
                        num_steps = max(1, int(num_interpolate_points))
                        for step in range(0, num_steps + 1):
                            t = step / num_steps
                            q_bridge = interpolate_motion(current_pose, q_start_orbit, t)
                            bridge_scene_state = entry_scene_state if t < 1.0 else start_scene_state
                            _apply_scene(
                                q_bridge,
                                bridge_scene_state["attachments"],
                                bridge_scene_state["grasp_poses"],
                                bridge_scene_state["placement_poses"],
                            )

                            p.stepSimulation(physicsClientId=self.using_client)
                            time.sleep(delay)

                if len(configs) == 1:
                    config = configs[0]
                    _apply_scene(
                        config.joint_angles,
                        start_scene_state["attachments"],
                        start_scene_state["grasp_poses"],
                        start_scene_state["placement_poses"],
                    )

                    p.stepSimulation(physicsClientId=self.using_client)
                    time.sleep(delay)

                    current_pose = config.joint_angles
                    current_scene_state = start_scene_state
                    continue

                # Motion between configurations within the current orbit
                for k in range(len(configs) - 1):
                    config1 = configs[k]
                    config2 = configs[k + 1]
                    config1_scene_state = _resolve_scene_state(
                        config1,
                        orbit_attachments,
                        orbit_grasp_poses,
                        orbit_placement_poses,
                    )
                    config2_scene_state = _resolve_scene_state(
                        config2,
                        orbit_attachments,
                        orbit_grasp_poses,
                        orbit_placement_poses,
                    )
                    

                    num_steps = max(1, int(num_interpolate_points))
                    # Debug mode: disable interpolation and replay configuration endpoints only.
                    # num_steps = 1
                    
                    for step in range(0, num_steps + 1):
                        t = step / num_steps
                        q_sample = interpolate_motion(config1.joint_angles, config2.joint_angles, t)
                        
                        # For debugging: print joint angles at each step
                        # print(f"Step {step}/{num_steps}: t={t:.2f}, q_sample={q_sample}")


                        active_scene_state = config1_scene_state if t < 1.0 else config2_scene_state
                        _apply_scene(
                            q_sample,
                            active_scene_state["attachments"],
                            active_scene_state["grasp_poses"],
                            active_scene_state["placement_poses"],
                        )

                        p.stepSimulation(physicsClientId=self.using_client)
                        time.sleep(delay)

                    _hold_transition(
                        config2,
                        config1_scene_state,
                        config2_scene_state,
                    )

                current_pose = configs[-1].joint_angles
                current_scene_state = _resolve_scene_state(
                    configs[-1],
                    orbit_attachments,
                    orbit_grasp_poses,
                    orbit_placement_poses,
                )
        
        
        finally:
            
            for r_id in range(self.num_robots):
                robot_sim_id = self.robots[r_id].sim_id
                p.resetJointState(robot_sim_id, 9, 0.04, physicsClientId=self.using_client)
                p.resetJointState(robot_sim_id, 10, 0.04, physicsClientId=self.using_client)

            time.sleep(2)
            p.stopStateLogging(log_id, physicsClientId=self.using_client)

        if not os.path.exists(output_path):
            print(f"Video logging finished but file was not created: {output_path}")
            return None
        if os.path.getsize(output_path) == 0:
            print(f"Video file is empty: {output_path}")
            return None

        # Sleep at the end to allow GUI and logger to finish
        print(f"Saved video: {output_path}")
        return output_path

    

    def close(self):
        """Disconnect active PyBullet clients safely."""
        if self.debugger_attached:
            print("Debugger is attached. Skip automatic disconnect to keep GUI alive.")
            return
        p.disconnect(self.using_client)

    def keep_gui_alive(self, step_dt=1.0 / 240.0):
        """Keep GUI responsive for debugging/inspection until interrupted."""
        if self.using_client is None or not p.isConnected(self.using_client):
            print("Visualization client is not connected.")
            return

        print("Keeping PyBullet GUI open. Press Ctrl+C to exit hold mode.")
        try:
            while p.isConnected(self.using_client):
                p.stepSimulation(physicsClientId=self.using_client)
                time.sleep(step_dt)
        except KeyboardInterrupt:
            print("Leaving GUI hold mode.")


####################################################################
#=================== MOTION PLANNING UTILS =======================
####################################################################


def sample_placement_pose(
    obj_id,
    T_grasp,
    x_goal,
    y_goal,
    z_goal,
    environment,
    r_id,
    orbit=None,
    pre_pose_offset=0.1,
    yaw_discretization=1,
    z_offset=0.01,
):
    
    success = False

    for _ in range(NUM_TRIAL_SAMPLES):
        target_orn = sample_yaw_quaternion(
            discretization=yaw_discretization
        )  # Sample an upright object yaw.

        T_obj_placement = make_transform(           # object pose in world frame
            [x_goal, y_goal, z_goal + z_offset],
            target_orn
        )

        # For placing
        T_ee_place = T_obj_placement @ np.linalg.inv(T_grasp)

        # Pre placing pose
        T_approach = np.eye(4)
        T_approach[:3, 3] = np.array([0, 0, -pre_pose_offset])  # Move along the z-axis of the end-effector frame
        T_ee_pre_place = T_ee_place @ T_approach

        # Calling IK
        r_joint_angles = environment.robots[r_id].inverse_kinematics_from_ee_pose(T_ee_place)
        r_joint_angles_pre = environment.robots[r_id].inverse_kinematics_from_ee_pose(T_ee_pre_place)

        # Check feasibility
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
    

def sample_grasp_pose(
    obj_id,
    environment,
    r_id,
    orbit,
    pre_pose_offset=0.1,
    discretization=1,
    grasp_offset_z=GRASP_OFFSET_Z,
):
    success = False
    # print(f'Robot {r_id} is trying to grasp object {obj_id}')

    for _ in range(NUM_TRIAL_SAMPLES):

        # TOP GRASP: the coordinate of gripper is same with object -> roll pi only
        target_orn = sample_yaw_quaternion(roll_target=np.pi, pitch_target=0, discretization=discretization)     # grasping from top only T_obj = T_ee @ T_grasp
        T_grasp = make_transform(
            [GRASP_OFFSET_X, GRASP_OFFSET_Y, grasp_offset_z],
            target_orn,
        ) # EE to object transform (object pose in EE frame)

        T_object = orbit.placement_poses[obj_id]
        
        # For grasping
        T_ee_grasp = T_object @ np.linalg.inv(T_grasp)

        # Pre pose of grasping pose
        T_approach = np.eye(4)
        T_approach[:3, 3] = np.array([0, 0, -pre_pose_offset])  # Move along the z-axis of the end-effector frame
        T_ee_pre_grasp = T_ee_grasp @ T_approach
        
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



def make_transform(position, orientation):
    ''' Create a homogeneous transformation matrix from position and orientation '''
    rot_matrix = R.from_quat(orientation).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = position
    return transform

def rotation_matrix_to_quaternion(rot_matrix):
    """Convert a 3x3 rotation matrix to quaternion [x, y, z, w]."""
    return R.from_matrix(np.asarray(rot_matrix, dtype=float)).as_quat().tolist()



def sample_yaw_quaternion(roll_target=0.0, pitch_target=0.0, discretization=1): # pick from top only
    # Sample a random yaw angle and convert it to a quaternion
    yaw_candidates = np.linspace(-np.pi, np.pi, discretization, endpoint=False)
    yaw = random.choice(yaw_candidates)

    rot = R.from_euler(
        'xyz',
        [roll_target, pitch_target, yaw]
    )

    return rot.as_quat().tolist()





if __name__ == "__main__":
    env = SimplePickPlaceEnv(num_robots=2, num_mobjects=2)
    try:
        env.keep_gui_alive()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if sys.gettrace() is None:
            env.close()
        else:
            print("Debugger mode: leaving clients connected on exit.")
