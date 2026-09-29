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
from env.simple_pick_place import FR3_LIMITS, SimplePickPlaceEnv, FrankaPandaRobot, MovableObject



NUM_TRIAL_SAMPLES = 10
GRASP_OFFSET_X = 0.0    # distance from end-effector to object when grasping, along the vertical direction (z-axis in EE frame)
GRASP_OFFSET_Y = 0.0
GRASP_OFFSET_Z = 0.02
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



class HandoverEnv(SimplePickPlaceEnv):
    """
        The environment of two robots performing a handover task.
        Using robot: Franka Panda

        Searching space for robot: 
        [
            [joint1_r1, joint2_r1, ..., joint7_r1], # Robot 1 state
            [joint1_r2, joint2_r2, ..., joint7_r2]  # Robot 2 state
        ]

    """

    def __init__(self, num_robots=2, num_mobjects=1, show_gui=True):

        if show_gui:
            self.using_client = p.connect(p.GUI)
        else:
            self.using_client = p.connect(p.DIRECT)

        self.num_robots = num_robots
        self.num_mobjects = num_mobjects

        self.r_min = 1.0        # PRM lower bound connection threshold
        self.r_prm = 18.0       # PRM base scaling factor (gamma) for maximum initial search radius
        self.k_prm = 7              # PRM number of nearest neighbors to connect to

        # Keep the collision-checking interface consistent with the current
        # SimplePickPlaceEnv and dRRT* constituent/tensor roadmap code.
        self.collision_checking_resolution = 0.5
        self.collision_joint_resolution = 0.2

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

        self.config_dim = num_robots * 7    # 7 joint angles for each robot
        self.configuration_type_radian = True
        self.configuration_type_cartesian = False

        self.joint_limits = FR3_LIMITS
        self.debugger_attached = sys.gettrace() is not None

        self.reset(client_id=self.using_client)

        # For checking time of collision checking and IK solving
        self.profile_time = defaultdict(float)
        self.profile_calls = defaultdict(int)

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
            "models/custom_table/custom_table.urdf", 
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
        offset_y = [0.5, -0.5]  
        for i in range(self.num_robots):
            rid = i
            offset_y_i = offset_y[i % len(offset_y)]
            offset_x_i = -0.2
            robot_base_pos = [table1_pos[0] + offset_x_i, table1_pos[1] + offset_y_i, table_surface_z]
            robot = FrankaPandaRobot(id=rid, base_pos=robot_base_pos, client_id=self.using_client)
            self.robots.append(robot)

            self.init_attachments[rid] = None                   # No attachments at the start 
            self.init_robot_states[rid] = RobotState.TRANSIT    # All robots start in TRANSIT state


        # Create multiple movable objects
        self.mobjects = []
        for i in range(self.num_mobjects):
            oid = i
            offset_x = 0.15+0.15*i
            offset_y = -0.5
            offset_z = 0.025   # from the center of the cube to the table surface
            obj_pos = [table1_pos[0] + offset_x, table1_pos[1] + offset_y, table_surface_z + offset_z]  # on the table surface
            obj_orientation = p.getQuaternionFromEuler([0, 0, 0])   # no rotation

            offset_x_goal = offset_x
            offset_y_goal = 0.35
            target_pos = [table1_pos[0] + offset_x_goal, table1_pos[1] + 0.3 + offset_y_goal, table_surface_z + offset_z]  

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

            obj_target = [obj.x_goal, obj.y_goal, obj.z_goal]

            print("Created one movable object with id {}, initial position {}, and goal regions {}".format(
                obj.id, [obj.x, obj.y, obj.z], obj_target
            ))


        self._draw_object_targets(client_id=self.using_client)  # Visualize target objects in the same client

        # Turn of all the coordinates
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        # # FOR CHECKING GRASPING POSES
        # state = p.getLinkState(self.robots[0].sim_id, self.robots[0].ee_link_index, physicsClientId=self.using_client)
        # pos = state[4]
        # orn = state[5]
        # print(f"EE pose: pos {pos}, orn {orn}")
        # self.draw_frame(pos, orn)

        # Plot the base pose of robot 1.
        # robot_1_id = self.robots[1].sim_id
        # pos, orn = p.getBasePositionAndOrientation(
        #     robot_1_id,
        #     physicsClientId=self.using_client,
        # )
        # print(f"Robot 1 base pose: pos {pos}, orn {orn}")
        # self.draw_frame(pos, orn)

        # obj_state = p.getBasePositionAndOrientation(self.mobjects[0].sim_id, physicsClientId=self.using_client)
        # obj_pos = obj_state[0]
        # obj_orn = obj_state[1]
        # self.draw_frame(obj_pos, obj_orn)


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
            if not np.allclose(current_pose[:3, 3], target_pos, atol=0.02): # 2cm tolerance
                return False

        for r_id, target_obj_id in goal_attachments.items():
            actual_obj_id = orbit.attachments.get(r_id)
            if actual_obj_id != target_obj_id:
                return False

        orbit.is_goal_value = True
        return True


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
            angle = round(float(random.uniform(low, high)), 8)
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
        if not self.is_feasible_in_orbit(sampled_configuration, orbit):
            return None
        return sampled_configuration
    

    def sample_boundary_configuration(self, orbit):
        '''
             Sample single configuration that lies on the boundary of this orbit.
        '''

        # choose the subset of robots \Re
        transition_robots_id = random.choice(self.transition_robot_subsets)

        # available objects for pick action
        available_objects_id = list(range(self.num_mobjects))
        for grasping_obj_id in orbit.attachments.values():
            if grasping_obj_id in available_objects_id:
                available_objects_id.remove(grasping_obj_id)

        # Sample configuration for all robots
        num_robots = len(self.robots)
        joint_angles = [None] * num_robots
        pre_pose = [None] * num_robots
        new_attachments = copy.deepcopy(orbit.attachments)
        new_grasp_poses = {}
        new_placement_poses = {}

        handover_pairs = {}  # (robot_id, robot_id) for handover actions
        free_robots = [r_id for r_id in self.robot_list if r_id not in orbit.attachments or orbit.attachments[r_id] is None]

        for r_id in self.robot_list:
            
            if r_id in handover_pairs.values() or r_id in handover_pairs.keys():
                continue    # already sampled for this robot in handover action, skip

            if r_id in transition_robots_id:   # this robot is doing transition action
                # --------------------------
                # PICK ACTION
                # --------------------------
                if orbit.attachments[r_id] is None:  
                    if len(available_objects_id) == 0:
                        # No available objects to pick, this robot cannot do pick action, skip pick but random a joint angle
                        joint_angles[r_id] = self.random_joint_angles_for_robot(r_id)
                        pre_pose[r_id] = joint_angles[r_id].copy()
                        continue
                    else:
                        obj_id = random.choice(available_objects_id)
                        r_joint_angles, r_joint_angles_pre, T_grasp = sample_grasp_pose(
                            obj_id, self, r_id, orbit
                        )
                        if (
                            T_grasp is None
                            or r_joint_angles is None
                            or r_joint_angles_pre is None
                        ):
                            # robot cannot find feasible grasp pose for this object, skip pick but random a joint angle
                            joint_angles[r_id] = self.random_joint_angles_for_robot(r_id)
                            pre_pose[r_id] = joint_angles[r_id].copy()
                            continue
                        r_joint_angles = np.round(r_joint_angles, 4).tolist()
                        r_joint_angles_pre = np.round(
                            r_joint_angles_pre, 4
                        ).tolist()
                        
                        # Update the values
                        joint_angles[r_id] = r_joint_angles
                        pre_pose[r_id] = r_joint_angles_pre
                        new_grasp_poses[obj_id] = T_grasp
                        new_attachments[r_id] = obj_id
                        available_objects_id.remove(obj_id)
                        free_robots.remove(r_id)


                else:  
                    will_handover = random.choice([True, False])
                    if len(free_robots) > 0 and will_handover:
                        # --------------------------
                        # HANDOVER ACTION
                        # --------------------------
                        holding_robot_id = r_id
                        grasping_obj_id = orbit.attachments[holding_robot_id]
                        grasping_pose = orbit.grasp_poses[grasping_obj_id]
                        other_robot_id = random.choice(free_robots)

                        # sample the pose of the handover for both two robots
                        (
                            q1,
                            q1_pre,
                            q2,
                            q2_pre,
                            T_new_grasp,
                        ) = sample_handover_pose(
                            self,
                            holding_robot_id,
                            other_robot_id,
                            grasping_pose,
                            orbit=orbit,
                        )
                        if (
                            T_new_grasp is None
                            or q1 is None
                            or q1_pre is None
                            or q2 is None
                            or q2_pre is None
                        ):
                            # cannot find feasible handover pose for this object, skip handover but random a joint angle for the current robot
                            joint_angles[holding_robot_id] = self.random_joint_angles_for_robot(holding_robot_id)
                            pre_pose[holding_robot_id] = joint_angles[
                                holding_robot_id
                            ].copy()
                            continue
                        q1 = np.round(q1, 4).tolist()
                        q1_pre = np.round(q1_pre, 4).tolist()
                        q2 = np.round(q2, 4).tolist()
                        q2_pre = np.round(q2_pre, 4).tolist()
                        # Update the values
                        joint_angles[holding_robot_id] = q1
                        joint_angles[other_robot_id] = q2
                        pre_pose[holding_robot_id] = q1_pre
                        pre_pose[other_robot_id] = q2_pre
                        new_grasp_poses[grasping_obj_id] = T_new_grasp
                        new_attachments[other_robot_id] = grasping_obj_id
                        new_attachments[holding_robot_id] = None
                        handover_pairs[holding_robot_id] = other_robot_id
                        free_robots.remove(other_robot_id)

                    else:
                        # --------------------------
                        # PLACE ACTION
                        # --------------------------
                        obj_id = orbit.attachments[r_id]
                        x_place = self.mobjects[obj_id].x_goal  # FIXME: this is ideal case
                        y_place = self.mobjects[obj_id].y_goal
                        z_place = self.mobjects[obj_id].z_goal
                        (
                            r_joint_angles,
                            r_joint_angles_pre,
                            T_object,
                        ) = sample_placement_pose(
                            obj_id,
                            orbit.grasp_poses[obj_id],
                            x_goal=x_place,
                            y_goal=y_place,
                            z_goal=z_place,
                            environment=self,
                            r_id=r_id,
                            orbit=orbit,
                        )
                        if (
                            T_object is None
                            or r_joint_angles is None
                            or r_joint_angles_pre is None
                        ):
                            # skip place but random a joint angle bcs robot can keep holding
                            joint_angles[r_id] = self.random_joint_angles_for_robot(r_id)
                            pre_pose[r_id] = joint_angles[r_id].copy()
                            continue
                        else:
                            r_joint_angles = np.round(r_joint_angles, 8).tolist()
                            r_joint_angles_pre = np.round(
                                r_joint_angles_pre, 8
                            ).tolist()
                            joint_angles[r_id] = r_joint_angles
                            pre_pose[r_id] = r_joint_angles_pre
                            new_placement_poses[obj_id] = T_object
                            new_attachments[r_id] = None
                            # this robot will not be free until the next step, no add this to free_robots list in this loop

            else:
                # this robot is not doing transition action, just sample random joint angles
                joint_angles[r_id] = self.random_joint_angles_for_robot(r_id)
                pre_pose[r_id] = joint_angles[r_id].copy()

        # Check if this is feasible
        if len(new_grasp_poses) + len(new_placement_poses) == 0:
            return None   # no actual transition action sampled, skip
        if any(q is None for q in joint_angles) or any(q is None for q in pre_pose):
            raise ValueError(
                "Every robot must have joint angles and a pre-pose in a "
                "handover boundary configuration."
            )
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
        p.performCollisionDetection(physicsClientId=self.using_client)

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
            for c in _first_penetration_contacts(robot_sim_id, attached_obj_sim_id, max_distance=SAFE_APPROACH_DISTANCE):
                link_robot = c[3] if c[1] == robot_sim_id else c[4]
                if link_robot in fingertip_links_by_robot.get(robot_sim_id, set()):
                    continue
                _debug_collision(c, prefix="Forbidden attachment collision")
                return True

        # Layer 2b: movable objects (placed and attached) with fixed objects.
        for obj_sim_id in movable_obj_ids:
            for fixed_sim_id in fixed_object_ids:
                contacts = _first_penetration_contacts(obj_sim_id, fixed_sim_id, max_distance=SAFE_APPROACH_DISTANCE)
                for c in contacts:
                    if obj_sim_id in placed_ids and c[8] >= -ALLOWED_PLACEMENT_CONTACT_DISTANCE:
                        continue
                    if c[8] <= 0:
                        _debug_collision(c, prefix="Object-environment collision")
                        return True
            # with plane
            for c in _first_penetration_contacts(obj_sim_id, self.plane, max_distance=SAFE_APPROACH_DISTANCE):
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
                for c in _first_penetration_contacts(robot_ids[i], robot_ids[j], max_distance=SAFE_APPROACH_DISTANCE):
                    _debug_collision(c, prefix="Robot-robot collision")
                    return True

        # Layer 3b: robot with fixed objects, allowing base/table support contacts.
        for robot_sim_id in robot_ids:
            robot_base_links = set(self.robots[robot_ids.index(robot_sim_id)].base_link_index)
            for fixed_sim_id in fixed_object_ids:
                for c in _first_penetration_contacts(robot_sim_id, fixed_sim_id, max_distance=SAFE_APPROACH_DISTANCE):
                    link_robot = c[3] if c[1] == robot_sim_id else c[4]
                    if link_robot in robot_base_links: 
                        continue
                    _debug_collision(c, prefix="Robot-environment collision")
                    return True
            # robot with plane
            for c in _first_penetration_contacts(robot_sim_id, self.plane, max_distance=SAFE_APPROACH_DISTANCE):
                _debug_collision(c, prefix="Robot-plane collision")
                return True

        # Layer 3c: robot with movable objects.
        for robot_sim_id in robot_ids:
            owner_obj_sim_id = attached_pairs.get(robot_sim_id)
            for obj_sim_id in movable_obj_ids:
                for c in _first_penetration_contacts(robot_sim_id, obj_sim_id, max_distance=SAFE_APPROACH_DISTANCE):
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

        # Extract the values
        t_ee = self.robots[robot_id].forward_kinematics(robot_joint_angles)
        for gripper_joint_id in (9, 10):
            p.resetJointState(
                self.robots[robot_id].sim_id,
                gripper_joint_id,
                0.0,
                physicsClientId=self.using_client,
            )
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
            contacts = p.getClosestPoints(obj_sim_id, current_object_sim_id, distance=SAFE_APPROACH_DISTANCE, physicsClientId=self.using_client)
            for c in contacts:
                if c[8] >= SAFE_APPROACH_DISTANCE:
                    continue
                return False
        # Collision checking between placing objects and fixed environment objects
        for fixed_sim_id in self.fixed_objects:
            contacts = p.getClosestPoints(current_object_sim_id, fixed_sim_id, distance=SAFE_APPROACH_DISTANCE, physicsClientId=self.using_client)
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

        for gripper_joint_id in (9, 10):
            p.resetJointState(
                self.robots[robot_id].sim_id,
                gripper_joint_id,
                0.04,
                physicsClientId=self.using_client,
            )
        
        # TODO: IK reachability check: if robot cannot reach the grasp pose, then it's not feasible
        
        # Robot's hand doesn't smash into the environment
        base_links = set(self.robots[robot_id].base_link_index)
        for fixed_sim_id in self.fixed_objects:
            contacts = p.getClosestPoints(
                robot_sim_id,
                fixed_sim_id,
                distance=SAFE_APPROACH_DISTANCE,
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
    

    def is_feasible_in_orbit(self, config, orbit):
        """Check if the given configuration is feasible in the context of the given orbit (considering attachments, grasp/placement poses)"""

        joint_angles = (
            config.joint_angles
            if hasattr(config, "joint_angles")
            else config
        )

        # move robot to the test configuration
        t_ee_poses = {}
        for r_id in range(self.num_robots):
            q_sample = joint_angles[r_id]
            if self.is_self_colliding(r_id, q_sample):
                return False
            # is_self_colliding already reset the robot to q_sample.
            gripper_target = (
                0.0 if orbit.attachments.get(r_id) is not None else 0.04
            )
            for gripper_joint_id in (9, 10):
                p.resetJointState(
                    self.robots[r_id].sim_id,
                    gripper_joint_id,
                    gripper_target,
                    physicsClientId=self.using_client,
                )
            t_ee_poses[r_id] = self.robots[r_id].get_end_effector_pose()

        # move objects to the test configuration
        placed_objects = []
        placed_objects_sim_ids = []
        objects_placements = orbit.placement_poses if orbit else None
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
                # print(f"Input config for Robot{robot_id} joint angles: {config.joint_angles[robot_id]}")
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
            # At a handover boundary the giver (old attachment) and receiver
            # (new attachment) both retain contact with the object.
            gripper_target = (
                0.0
                if (
                    orbit.attachments.get(r_id) is not None
                    or configuration.attachments.get(r_id) is not None
                )
                else 0.04
            )
            for gripper_joint_id in (9, 10):
                p.resetJointState(
                    self.robots[r_id].sim_id,
                    gripper_joint_id,
                    gripper_target,
                    physicsClientId=self.using_client,
                )
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
                handing_robot_id = None
                holding_robot_id = None
                for r_id, holding_object in configuration.attachments.items():
                    if holding_object == object_id:
                        holding_robot_id = r_id
                        if object_id in orbit.attachments.values():
                            handing_robot_id = [r_id for r_id, obj_id in orbit.attachments.items() 
                                                if obj_id == object_id][0]
                        break
                if holding_robot_id is not None:
                    t_object = t_ee_poses[holding_robot_id] @ state['pose'] 
                    self.mobjects[object_id].set_pose(t_object)
                    attachment_sim_ids[self.robots[holding_robot_id].sim_id] = self.mobjects[object_id].sim_id
                # when grasping, at the boundary point the object still touching the table
                placed_objects_sim_ids.append(self.mobjects[object_id].sim_id)
                # at the handover one object can be held by two robots at the same time
                if handing_robot_id is not None and handing_robot_id != holding_robot_id:
                    attachment_sim_ids[self.robots[handing_robot_id].sim_id] = self.mobjects[object_id].sim_id

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


    def collision_free_motion(self, config1, config2, orbit=None, resolution=0.05, num_checks=None):
        """Check straight-line motion using adaptive midpoint checks until neighboring joint samples are within resolution."""
        if resolution <= 0:
            raise ValueError("resolution must be > 0")

        q_start = np.ravel(config1.joint_angles)
        q_goal = np.ravel(config2.joint_angles)
        total_motion = np.linalg.norm(q_goal - q_start)

        # Backward compatibility: map legacy num_checks to joint-space step length.
        if num_checks is not None:
            if num_checks <= 0:
                return True
            resolution = min(resolution, total_motion / float(num_checks))

        # Start from the full segment and bisect while neighboring samples are too far apart.
        intervals = [(0.0, 1.0, config1.joint_angles, config2.joint_angles)]

        while intervals:
            t_left, t_right, q_left, q_right = intervals.pop()
            step_distance = np.linalg.norm(np.ravel(q_right) - np.ravel(q_left))
            if step_distance <= resolution:
                continue

            t_mid = 0.5 * (t_left + t_right)
            q_step = interpolate_motion(config1.joint_angles, config2.joint_angles, t_mid)
            interp_config = Configuration(
                joint_angles=q_step,
                attachments=config1.attachments,  # or config2.attachments, they should be the same
                is_transition=False,
                parent_orbit=orbit,
            )
            if not self.is_feasible_in_orbit(interp_config, orbit):
                return False

            intervals.append((t_left, t_mid, q_left, q_step))
            intervals.append((t_mid, t_right, q_step, q_right))

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
            client_id = self.using_client
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
        line_width=3

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


    def simulate_final_path(self, output_path_filename, output_sim_filename="simulation.mp4", delay=0.2, num_interpolate_points=10):
        """Load an orbit-structured planned path text file and simulate it in the GUI while recording an MP4."""
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

            def _resolve_scene_state(config, orbit_attachments, orbit_grasp_poses, orbit_placement_poses):
                attachments = dict(orbit_attachments)
                attachments.update(config.attachments or {})

                grasp_poses = dict(orbit_grasp_poses)
                placement_poses = dict(orbit_placement_poses)

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

                # Motion between the final configuration of previous orbit and the initial configuration of current orbit.
                if current_pose is not None:
                    q_start_orbit = configs[0].joint_angles
                    bridge_motion = np.linalg.norm(np.ravel(current_pose) - np.ravel(q_start_orbit))
                    if bridge_motion > 1e-8:
                        print("Transition between orbits")
                        num_steps = max(1, int(num_interpolate_points))
                        for step in range(0, num_steps + 1):
                            t = step / num_steps
                            q_bridge = interpolate_motion(current_pose, q_start_orbit, t)
                            _apply_scene(
                                q_bridge,
                                start_scene_state["attachments"],
                                start_scene_state["grasp_poses"],
                                start_scene_state["placement_poses"],
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
                    continue
                
                print(f"Simulating {len(configs)} configurations")
                

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
                    
                    print(f"Transition from config {k} to {k+1}")

                    num_steps = max(1, int(num_interpolate_points))
                    # Debug mode: disable interpolation and replay configuration endpoints only.
                    # num_steps = 1
                    
                    for step in range(0, num_steps + 1):
                        t = step / num_steps
                        q_sample = interpolate_motion(config1.joint_angles, config2.joint_angles, t)
                        
                        # For debugging: print joint angles at each step
                        print(f"Step {step}/{num_steps}: t={t:.2f}, q_sample={q_sample}")


                        active_scene_state = config1_scene_state if t < 1.0 else config2_scene_state
                        _apply_scene(
                            q_sample,
                            active_scene_state["attachments"],
                            active_scene_state["grasp_poses"],
                            active_scene_state["placement_poses"],
                        )

                        p.stepSimulation(physicsClientId=self.using_client)
                        time.sleep(delay)

                current_pose = configs[-1].joint_angles
        
        
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
def inverse_kinematics(robot_id, target_pos, target_orn, environment):
    """Compute the inverse kinematics for the given target end-effector pose and robot_id
    
        :param target_pos: Target end-effector position as a list of 3 values [x, y, z]
        :param target_orn: Target end-effector orientation as a quaternion [x, y, z, w]
        :param robot_id: ID of the robot to compute IK for
        :param environment: The environment object containing the robots
        :return: Joint angles as a list of 7 values, or None if no solution is found
    """
    # Find the target robot
    target_robot = next((r for r in environment.robots if r.id == robot_id), None)
    if target_robot is None:
        return None
    return target_robot.inverse_kinematics(target_pos, target_orn)
    

def inverse_kinematics_from_pose(robot_id, target_pose, environment):
    """Compute the inverse kinematics for the given target end-effector pose and robot_id
    
        :param target_pose: Target end-effector pose as a 4x4 numpy homogeneous transformation matrix
        :param robot_id: ID of the robot to compute IK for
        :param environment: The environment object containing the robots
        :return: Joint angles as a list of 7 values, or None if no solution is found
    """
    target_pos = target_pose[:3, 3]
    rot_matrix = target_pose[:3, :3]
    target_orn = rotation_matrix_to_quaternion(rot_matrix)

    return inverse_kinematics(robot_id, target_pos, target_orn, environment)


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
):
    
    success = False        
    z_offset = 0.02                             # add a small offset to prevent collision with the table

    for _ in range(NUM_TRIAL_SAMPLES):
        target_orn = sample_yaw_quaternion(roll_target=np.pi, pitch_target=0)    # sample object orientation (yaw only, upright) -> parallel to table surface.

        T_obj_placement = make_transform(           # object pose in world frame
            [x_goal, y_goal, z_goal + z_offset],
            target_orn
        )
        T_ee = T_obj_placement @ np.linalg.inv(T_grasp)

        T_approach = np.eye(4)
        T_approach[:3, 3] = np.array([0, 0, -pre_pose_offset])
        T_ee_pre = T_ee @ T_approach

        r_joint_angles = inverse_kinematics_from_pose(r_id, T_ee, environment)
        r_joint_angles_pre = inverse_kinematics_from_pose(
            r_id, T_ee_pre, environment
        )
        if r_joint_angles is None or r_joint_angles_pre is None:
            continue
        if not environment.is_placement_collision_free(obj_id, r_id, r_joint_angles, orbit):         # Broadphase collision check
            continue
        if not environment.is_pre_pose_feasible(
            r_id, r_joint_angles_pre, r_joint_angles, orbit
        ):
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
):
    success = False
    # print(f'Robot {r_id} is trying to grasp object {obj_id}')

    for _ in range(NUM_TRIAL_SAMPLES):

        # TOP GRASP: the coordinate of gripper is same with object -> roll pi only
        target_orn = sample_yaw_quaternion(roll_target=np.pi, pitch_target=0)     # grasping from top only T_obj = T_ee @ T_grasp
        T_grasp = make_transform([GRASP_OFFSET_X, GRASP_OFFSET_Y, GRASP_OFFSET_Z], target_orn) # EE to object transform (object pose in EE frame)

        T_object = orbit.placement_poses[obj_id]
        T_ee = T_object @ np.linalg.inv(T_grasp)

        T_approach = np.eye(4)
        T_approach[:3, 3] = np.array([0, 0, -pre_pose_offset])
        T_ee_pre = T_ee @ T_approach

        r_joint_angles = inverse_kinematics_from_pose(r_id, T_ee, environment)
        r_joint_angles_pre = inverse_kinematics_from_pose(
            r_id, T_ee_pre, environment
        )
        if r_joint_angles is None or r_joint_angles_pre is None:
            continue
        if not environment.is_grasp_feasible(obj_id, r_id, r_joint_angles, orbit):         # Broadphase collision check
            continue
        if not environment.is_pre_pose_feasible(
            r_id, r_joint_angles_pre, r_joint_angles, orbit
        ):
            continue
        success = True
        break
    
    if success:
        return r_joint_angles, r_joint_angles_pre, T_grasp
    else:
        return None, None, None


def sample_handover_pose(
    environment,
    giver_robot_id,
    receiver_robot_id,
    T_grasp,
    orbit=None,
    pre_pose_offset=0.1,
):

    success = False
    
    for attempt in range(NUM_TRIAL_SAMPLES):

        # Sample a random point within the shared workspace bubble
        r1_base = environment.robots[giver_robot_id].base_pos
        r2_base = environment.robots[receiver_robot_id].base_pos
        object_xyz = sample_in_shared_workspace(r1_base, r2_base)
        if object_xyz is None:
            continue  # No overlapping workspace, try again
        
        object_ori = R.from_euler('x', np.pi/2).as_quat().tolist()
        T_object = make_transform(object_xyz, object_ori)
        
        # IK for Robot 1
        giver_robot_ee_pose = T_object @ np.linalg.inv(T_grasp)         # Desired EE pose for the giver robot to hold the object
        q1 = environment.robots[giver_robot_id].inverse_kinematics_from_ee_pose(giver_robot_ee_pose)
        if q1 is None or environment.robots[giver_robot_id].is_self_colliding(q1):
            continue  

        # calculate receiver robot ee pose for the handover
        T_opposite_grasp = calculate_opposing_grasp(T_grasp)
        r2_target_ee_pose = T_object @ np.linalg.inv(T_opposite_grasp)  # Desired EE pose for the receiver robot to grasp the object from the opposite side
        
        # IK for Robot 2
        q2 = environment.robots[receiver_robot_id].inverse_kinematics_from_ee_pose(r2_target_ee_pose)
        if q2 is None or environment.robots[receiver_robot_id].is_self_colliding(q2):
            continue

        # Unlike table grasp/place approaches, a handover is approached
        # laterally along the global Y axis. Each robot retreats toward its
        # own base, placing the two pre-poses on opposite sides of the shared
        # handover configuration.
        def global_y_pre_pose(ee_pose, robot_id):
            pre_pose = np.array(ee_pose, dtype=float, copy=True)
            base_y = float(environment.robots[robot_id].base_pos[1])
            handover_y = float(T_object[1, 3])
            direction = np.sign(base_y - handover_y)
            if direction == 0.0:
                other_robot_id = (
                    receiver_robot_id
                    if robot_id == giver_robot_id
                    else giver_robot_id
                )
                other_base_y = float(
                    environment.robots[other_robot_id].base_pos[1]
                )
                direction = 1.0 if base_y >= other_base_y else -1.0
            pre_pose[1, 3] += direction * pre_pose_offset
            return pre_pose

        giver_pre_pose = global_y_pre_pose(
            giver_robot_ee_pose, giver_robot_id
        )
        receiver_pre_pose = global_y_pre_pose(
            r2_target_ee_pose, receiver_robot_id
        )

        q1_pre = environment.robots[
            giver_robot_id
        ].inverse_kinematics_from_ee_pose(giver_pre_pose)
        q2_pre = environment.robots[
            receiver_robot_id
        ].inverse_kinematics_from_ee_pose(receiver_pre_pose)
        if q1_pre is None or q2_pre is None:
            continue
        if environment.robots[giver_robot_id].is_self_colliding(q1_pre):
            continue
        if environment.robots[receiver_robot_id].is_self_colliding(q2_pre):
            continue

        if orbit is not None:
            if not environment.is_pre_pose_feasible(
                giver_robot_id, q1_pre, q1, orbit
            ):
                continue
            if not environment.is_pre_pose_feasible(
                receiver_robot_id, q2_pre, q2, orbit
            ):
                continue

            # Validate the two lateral pre-poses together in the source orbit,
            # including the object still carried by the giver.
            pre_configuration = Configuration(
                joint_angles=[
                    q1_pre if robot_id == giver_robot_id
                    else q2_pre if robot_id == receiver_robot_id
                    else environment.robots[robot_id].init_joint_angles
                    for robot_id in environment.robot_list
                ],
                attachments=orbit.attachments,
                is_transition=False,
                parent_orbit=orbit,
            )
            if not environment.is_feasible_in_orbit(
                pre_configuration, orbit
            ):
                continue

        success = True
        break

    if success:
        return q1, q1_pre, q2, q2_pre, T_opposite_grasp
    else:    
        return None, None, None, None, None


def calculate_opposing_grasp(T_grasp):
    """
    Calculate the desired end-effector pose for the receiver robot to grasp the object from the opposite side.
    
    T_grasp: 4x4 homogeneous transformation matrix of the grasp pose relative to the object
    """
    
    # To grasp from the opposite side, we can rotate the grasp pose by 180 degrees around the vertical (z) axis in the object frame.
    R_opposite = R.from_euler('x', 180, degrees=True).as_matrix()
    # R_opposite = R.from_euler('z', 360, degrees=True).as_matrix()
    T_opposite_grasp = np.eye(4)
    T_opposite_grasp[:3, :3] = R_opposite @ T_grasp[:3, :3]     # Rotate the orientation
    T_opposite_grasp[:3, 3] = T_grasp[:3, 3]                    # Keep the same grasping offset

    return T_opposite_grasp
    


def make_transform(position, orientation):
    ''' 
        Create a homogeneous transformation matrix from position and orientation 
        position: list or array of 3 values [x, y, z]
        orientation: quaternion as list of 4 values [x, y, z, w]
        returns: 4x4 numpy array representing the homogeneous transformation matrix
    '''
    rot_matrix = R.from_quat(orientation).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = position
    return transform

def rotation_matrix_to_quaternion(rot_matrix):
    """Convert a 3x3 rotation matrix to quaternion [x, y, z, w]."""
    return R.from_matrix(np.asarray(rot_matrix, dtype=float)).as_quat().tolist()



def sample_yaw_quaternion(roll_target=0.0, pitch_target=0.0, discretization=4): # pick from top only
    # Sample a random yaw angle and convert it to a quaternion
    yaw_candidates = np.linspace(-np.pi, np.pi, discretization, endpoint=False)
    yaw = random.choice(yaw_candidates)

    rot = R.from_euler(
        'xyz',
        [roll_target, pitch_target, yaw]
    )

    return rot.as_quat().tolist()


def sample_in_shared_workspace(base1_pos, base2_pos):
    """
    Samples an object's (x, y, z) position within the overlapping 
    reachable workspaces of two Franka Emika robots.
    
    base1_pos, base2_pos: Tuples or lists of (x, y, z) for the robot bases.
    """
    # 1. Define physical reach constraints of a Franka Emika Panda arm (in meters)
    MAX_REACH = 0.85  # Maximum extension from base
    MIN_REACH = 0.15  # Minimum distance to avoid self-collision with the trunk

    # 2. Calculate the bounding box of the overlapping region to focus our samples
    # This prevents sampling randomly across the whole room
    min_x = max(base1_pos[0], base2_pos[0]) - MAX_REACH
    max_x = min(base1_pos[0], base2_pos[0]) + MAX_REACH
    if min_x > max_x:
        return None  # No overlap in x-axis reachable workspace
    
    min_y = max(base1_pos[1], base2_pos[1]) - MAX_REACH
    max_y = min(base1_pos[1], base2_pos[1]) + MAX_REACH
    if min_y > max_y:
        return None  # No overlap in y-axis reachable workspace
    
    # Handovers usually happen between the table height and shoulder height
    min_z = max(base1_pos[2], base2_pos[2]) + 0.10 # 10cm above base plate
    max_z = max(base1_pos[2], base2_pos[2]) + 0.60 # 60cm above base plate

    # 3. Rejection Sampling Loop
    for _ in range(NUM_TRIAL_SAMPLES*10):
        # Sample a random coordinate inside the focused bounding box
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        z = random.uniform(min_z, max_z)
        sample_pt = np.array([x, y, z])

        # 4. Compute Euclidean distances to both robot bases
        dist_to_r1 = np.linalg.norm(sample_pt - np.array(base1_pos))
        dist_to_r2 = np.linalg.norm(sample_pt - np.array(base2_pos))

        # 5. Check if the point falls inside BOTH spherical shells simultaneously
        r1_reachable = (MIN_REACH <= dist_to_r1 <= MAX_REACH)
        r2_reachable = (MIN_REACH <= dist_to_r2 <= MAX_REACH)

        if r1_reachable and r2_reachable:
            return [x, y, z] # Valid handover point found!

    # Fallback: If your robot bases are placed too far apart, the intersection might be empty
    return None
    # raise ValueError("No shared workspace found. Check if your robot bases are placed more than 1.7 meters apart!")


if __name__ == "__main__":
    env = HandoverEnv(num_robots=2, num_mobjects=2)
    # obs = env.reset()
    try:
        env.keep_gui_alive()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if sys.gettrace() is None:
            env.close()
        else:
            print("Debugger mode: leaving clients connected on exit.")
