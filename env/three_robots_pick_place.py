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

from env.simple_pick_place import (
    FR3_LIMITS,
    MovableObject,
    SimplePickPlaceEnv,
    rotation_matrix_to_quaternion,
    sample_grasp_pose,
    sample_placement_pose,
)


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

#------------------------------
#----- FOR DEBUGGING ONLY -----
print_debug = False
#----------- END ---------



class FrankaPandaRobot:
    def __init__(self, id, sim_id=None, base_pos=[0, 0, 0], base_ori=None, client_id=None):
        self.id = id
        self.sim_id = sim_id
        self.client_id = client_id
        self.base_pos = base_pos
        self.base_ori = (0.0, 0.0, 0.0, 1.0) if base_ori is None else tuple(base_ori)
        if len(self.base_ori) != 4:
            raise ValueError("base_ori must be a PyBullet quaternion [x, y, z, w]")
        self.init_joint_angles = [0, -0.5, 0, -2.5, 0, 2.0, 0.8]  # Neutral pose
        self.joint_limits = tuple(FR3_LIMITS)
        self.max_reach = 1.06

        if self.sim_id is None:
            # Create pybullet robot here and set self.sim_id
            self.sim_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=base_pos,
                baseOrientation=self.base_ori,
                useFixedBase=True,
                physicsClientId=self.client_id
            )
        
        # Testing panda link infor
        # for i in range(p.getNumJoints(self.sim_id)):
        #     info = p.getJointInfo(self.sim_id, i)
        #     print(f"Joint {i}: {info[1].decode()} | Link: {info[12].decode()}")

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
        """Set all movable arm joints, matching the shared robot interface."""
        if len(joint_angles) != len(self.movable_joint_indices):
            raise ValueError(
                f"Expected {len(self.movable_joint_indices)} joint angles, "
                f"got {len(joint_angles)}"
            )
        for joint_idx, angle in zip(self.movable_joint_indices, joint_angles):
            p.resetJointState(
                self.sim_id,
                joint_idx,
                float(angle),
                physicsClientId=self.client_id,
            )



class ThreeRobotsPickPlaceEnv(SimplePickPlaceEnv):
    """
        The environment of pick and place objects inside one table.
        Using robot: Franka Panda

        Searching space for robot: 
        [
            [joint1_r1, joint2_r1, ..., joint7_r1], # Robot 1 state
            [joint1_r2, joint2_r2, ..., joint7_r2]  # Robot 2 state
            [joint1_r3, joint2_r3, ..., joint7_r3]  # Robot 3 state
            [joint1_r4, joint2_r4, ..., joint7_r4]  # Robot 4 state
            ...

    """

    def __init__(self, num_robots=1, num_mobjects=1, show_gui=True):

        if show_gui:
            self.using_client = p.connect(p.GUI)
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
        table1_pos = [0.0, 0.0, 0.0]
        table = p.loadURDF(
            "models/big_table/big_table.urdf", 
            table1_pos, 
            p.getQuaternionFromEuler([0, 0, np.pi/2]),
            useFixedBase=True,
            physicsClientId=self.using_client
        )
        self.fixed_objects = [table]
        table_height = 0.88


        # Robot base — mounted on the table surface (table z + table height=0.625)
        self.robots = []
        table_surface_z = table1_pos[2] + table_height + 0.02  # robot base
        offset_y = [0.5, -0.5, 0.0]  # y offsets for multiple robots on the table
        # robot 0 and robot 1 as in our lab setup
        for i in range(min(2, self.num_robots)):
            rid = i
            offset_y_i = offset_y[i % len(offset_y)]
            offset_x_i = -0.6
            robot_base_pos = [table1_pos[0] + offset_x_i, table1_pos[1] + offset_y_i, table_surface_z]
            robot = FrankaPandaRobot(id=rid, base_pos=robot_base_pos, client_id=self.using_client)
            self.robots.append(robot)
            self.init_attachments[rid] = None                   # No attachments at the start 
            self.init_robot_states[rid] = RobotState.TRANSIT    # All robots start in TRANSIT state
        # robot 2 and robot 3 for more simulation experiments
        for i in range(2, self.num_robots):
            rid = i
            offset_y_i = offset_y[i % len(offset_y)]
            # Mount the second pair on the opposite side of the table.  Using
            # -0.2 here would place robots 2/3 directly on top of robots 0/1.
            offset_x_i = 0.6
            robot_base_pos = [table1_pos[0] + offset_x_i, table1_pos[1] + offset_y_i, table_surface_z]
            base_ori = p.getQuaternionFromEuler([0, 0, np.pi])  # Rotate the robot 180 degrees around the z-axis
            robot = FrankaPandaRobot(id=rid, base_pos=robot_base_pos, base_ori=base_ori, client_id=self.using_client)
            self.robots.append(robot)
            self.init_attachments[rid] = None                   # No attachments at the start 
            self.init_robot_states[rid] = RobotState.TRANSIT    # All robots start in TRANSIT state


        # Create multiple movable objects
        self.mobjects = []
        # Arrange objects in two lines across the center of the big table.
        # For eight objects this produces four objects per line:
        #   x=-0.15: y=[-0.525, -0.175, 0.175, 0.525]
        #   x= 0.15: y=[-0.525, -0.175, 0.175, 0.525]
        number_of_lines = 2
        objects_per_line = max(1, int(np.ceil(self.num_mobjects / number_of_lines)))
        object_spacing_y = 0.3
        line_x_positions = [-0.075, 0.075]
        goal_line_x_positions = [-0.2, 0.2]
        for i in range(self.num_mobjects):
            oid = i
            line_index = min(i // objects_per_line, number_of_lines - 1)
            column_index = i % objects_per_line
            offset_x = line_x_positions[line_index]
            offset_y = (
                column_index - (objects_per_line - 1) / 2.0
            ) * object_spacing_y
            # A target directly on robot 2's y=0 centerline produces a
            # graspable pose but no valid placement IK for the current
            # top-grasp family. Stagger only the center object on the
            # positive-X line by half a row spacing.
            if line_index == 1 and np.isclose(offset_y, 0.0):
                offset_y = object_spacing_y / 2.0
            offset_z = 0.025   # from the center of the cube to the table surface
            obj_pos = [table1_pos[0] + offset_x, table1_pos[1] + offset_y, table_surface_z + offset_z]  # on the table surface
            obj_orientation = p.getQuaternionFromEuler([0, 0, 0])   # no rotation
            offset_x_goal = goal_line_x_positions[line_index]
            offset_y_goal = offset_y
            offset_z_goal = offset_z
            target_pos = [table1_pos[0] + offset_x_goal, table1_pos[1] + offset_y_goal, table_surface_z + offset_z_goal]  
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

        # Turn of all the coordinates
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
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


    def sample_boundary_configuration(self, orbit):
        """Sample a boundary without picking objects already at their goal."""
        transition_robots_id = random.choice(self.transition_robot_subsets)

        attached_object_ids = {
            object_id
            for object_id in orbit.attachments.values()
            if object_id is not None
        }
        available_objects_id = [
            obj.id
            for obj in self.mobjects
            if obj.id not in attached_object_ids
            and not self._object_is_at_goal_in_orbit(obj.id, orbit)
        ]

        joint_angles = []
        pre_pose = []
        new_attachments = copy.deepcopy(orbit.attachments)
        new_grasp_poses = {}
        new_placement_poses = {}

        for r_id in self.robot_list:
            if r_id in transition_robots_id:
                if orbit.attachments[r_id] is None:
                    if not available_objects_id:
                        random_joint_angles = self.random_joint_angles_for_robot(r_id)
                        joint_angles.append(random_joint_angles)
                        pre_pose.append(random_joint_angles)
                        continue

                    obj_id = random.choice(available_objects_id)
                    (
                        r_joint_angles,
                        r_joint_angles_pre,
                        T_grasp,
                    ) = sample_grasp_pose(obj_id, self, r_id, orbit, discretization=2)
                    if (
                        T_grasp is None
                        or r_joint_angles is None
                        or r_joint_angles_pre is None
                    ):
                        random_joint_angles = self.random_joint_angles_for_robot(r_id)
                        joint_angles.append(random_joint_angles)
                        pre_pose.append(random_joint_angles)
                        continue

                    joint_angles.append(
                        np.round(r_joint_angles, 4).tolist()
                    )
                    pre_pose.append(
                        np.round(r_joint_angles_pre, 4).tolist()
                    )
                    new_grasp_poses[obj_id] = T_grasp
                    new_attachments[r_id] = obj_id
                    available_objects_id.remove(obj_id)
                else:
                    obj_id = orbit.attachments[r_id]
                    obj = self.mobjects[obj_id]
                    (
                        r_joint_angles,
                        r_joint_angles_pre,
                        T_object,
                    ) = sample_placement_pose(
                        obj_id,
                        orbit.grasp_poses[obj_id],
                        x_goal=obj.x_goal,
                        y_goal=obj.y_goal,
                        z_goal=obj.z_goal,
                        environment=self,
                        r_id=r_id,
                        orbit=orbit,
                    )
                    if (
                        T_object is None
                        or r_joint_angles is None
                        or r_joint_angles_pre is None
                    ):
                        random_joint_angles = self.random_joint_angles_for_robot(r_id)
                        joint_angles.append(random_joint_angles)
                        pre_pose.append(random_joint_angles)
                        continue

                    joint_angles.append(
                        np.round(r_joint_angles, 4).tolist()
                    )
                    pre_pose.append(
                        np.round(r_joint_angles_pre, 4).tolist()
                    )
                    new_placement_poses[obj_id] = T_object
                    new_attachments[r_id] = None
            else:
                random_joint_angles = self.random_joint_angles_for_robot(r_id)
                joint_angles.append(random_joint_angles)
                pre_pose.append(random_joint_angles)

        if not new_grasp_poses and not new_placement_poses:
            return None

        if (
            len(pre_pose) != self.num_robots
            or len(joint_angles) != self.num_robots
        ):
            raise ValueError(
                "Mismatch in number of robots and sampled joint angles or "
                "pre-poses."
            )

        sampled_configuration = Configuration(
            joint_angles=joint_angles,
            pre_pose=pre_pose,
            attachments=new_attachments,
            grasp_poses=new_grasp_poses,
            placement_poses=new_placement_poses,
            is_transition=True,
            parent_orbit=orbit,
        )
        if not self.is_feasible_at_boundary(sampled_configuration, orbit):
            return None
        return sampled_configuration


    def _object_is_at_goal_in_orbit(self, object_id, orbit):
        placement_pose = orbit.placement_poses.get(object_id)
        if placement_pose is None:
            return False

        return np.allclose(
            np.asarray(placement_pose, dtype=float)[:3, 3],
            self.mobjects[object_id].get_goal_position(),
            atol=0.02,
        )


    


if __name__ == "__main__":
    env = ThreeRobotsPickPlaceEnv(num_robots=3, num_mobjects=3, show_gui=True)
    try:
        env.keep_gui_alive()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if sys.gettrace() is None:
            env.close()
        else:
            print("Debugger mode: leaving clients connected on exit.")
