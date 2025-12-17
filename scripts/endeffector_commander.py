#!/usr/bin/env python3
"""
End-effector command conversion node
Convert end-effector target positions into joint commands
Uses a simplified inverse kinematics approach
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Header
from std_msgs.msg import Empty
from std_srvs.srv import Trigger, SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.optimize import minimize

class EndEffectorCommander(Node):
    def __init__(self):
        super().__init__('endeffector_commander')
        # Debug: sequence-based log alignment
        self.declare_parameter('debug_log_every_n', 30)
        self.declare_parameter('joint_state_rate', 10.0)  # Hz
        self.declare_parameter('position_change_threshold', 0.005)  # m
        self.debug_log_every_n = int(self.get_parameter('debug_log_every_n').value)
        self.joint_state_rate = float(self.get_parameter('joint_state_rate').value)
        self.position_change_threshold = float(self.get_parameter('position_change_threshold').value)
        
        # Subscribe to end-effector targets
        self.left_ee_sub = self.create_subscription(
            Float64MultiArray,
            '/left_arm/endeffector_target',
            self.left_ee_callback,
            10
        )
        self.right_ee_sub = self.create_subscription(
            Float64MultiArray,
            '/right_arm/endeffector_target',
            self.right_ee_callback,
            10
        )
        
        # Subscribe to gripper commands
        self.left_gripper_sub = self.create_subscription(
            Float64MultiArray,
            '/left_arm/gripper_command',
            self.left_gripper_callback,
            10
        )
        self.right_gripper_sub = self.create_subscription(
            Float64MultiArray,
            '/right_arm/gripper_command',
            self.right_gripper_callback,
            10
        )
        
        # Publish joint trajectory commands
        self.left_joint_pub = self.create_publisher(
            JointTrajectory,
            '/left_arm/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.right_joint_pub = self.create_publisher(
            JointTrajectory,
            '/right_arm/joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Publish joint states (for RViz visualization)
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        # Track all joint states
        self.all_joint_positions = {
            'left_shoulder_pan_joint': 0.0,
            'left_shoulder_lift_joint': 0.0,
            'left_elbow_joint': 0.0,
            'left_wrist_1_joint': -1.5708,
            'left_wrist_2_joint': 0.0,
            'left_wrist_3_joint': -3.1416,
            'right_shoulder_pan_joint': 0.0,
            'right_shoulder_lift_joint': 0.0,
            'right_elbow_joint': 0.0,
            'right_wrist_1_joint': -1.5708,
            'right_wrist_2_joint': 0.0,
            'right_wrist_3_joint': 3.1416,
            # Gripper joints (kept fixed)
            'left_robotiq_85_left_knuckle_joint': 0.0,
            'left_robotiq_85_right_knuckle_joint': 0.0,
            'left_robotiq_85_left_inner_knuckle_joint': 0.0,
            'left_robotiq_85_right_inner_knuckle_joint': 0.0,
            'left_robotiq_85_left_finger_tip_joint': 0.0,
            'left_robotiq_85_right_finger_tip_joint': 0.0,
            'right_robotiq_85_left_knuckle_joint': 0.0,
            'right_robotiq_85_right_knuckle_joint': 0.0,
            'right_robotiq_85_left_inner_knuckle_joint': 0.0,
            'right_robotiq_85_right_inner_knuckle_joint': 0.0,
            'right_robotiq_85_left_finger_tip_joint': 0.0,
            'right_robotiq_85_right_finger_tip_joint': 0.0,
        }
        
        # UR3e arm DH parameters (simplified model)
        self.d1 = 0.15185  # base height
        self.a2 = 0.24355  # upper arm length
        self.a3 = 0.21325  # forearm length
        self.d4 = 0.13105  # wrist 1 to end-effector
        self.d5 = 0.08535  # wrist 2 to end-effector
        self.d6 = 0.0921   # wrist 3 to end-effector
        
        # Joint limits (rad)
        self.left_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - fixed
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - fixed
            (-3.1417, -3.1415)  # wrist_3_joint (q6) - fixed
        ]
        self.right_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - fixed
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - fixed
            (3.1415, 3.1417)    # wrist_3_joint (q6) - fixed
        ]
        
        # Current joint states (for continuity)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e arm DH parameters (simplified model)
        self.d1 = 0.15185  # base height
        self.a2 = 0.24355  # upper arm length
        self.a3 = 0.21325  # forearm length
        self.d4 = 0.13105  # wrist 1 to end-effector
        self.d5 = 0.08535  # wrist 2 to end-effector
        self.d6 = 0.0921   # wrist 3 to end-effector
        
        # Joint limits (rad)
        self.left_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - fixed
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - fixed
            (-3.1417, -3.1415)  # wrist_3_joint (q6) - fixed
        ]
        self.right_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - fixed
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - fixed
            (3.1415, 3.1417)    # wrist_3_joint (q6) - fixed
        ]
        
        # Current joint states (for continuity)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e arm DH parameters (simplified model)
        self.d1 = 0.15185  # base height
        self.a2 = 0.24355  # upper arm length
        self.a3 = 0.21325  # forearm length
        self.d4 = 0.13105  # wrist 1 to end-effector
        self.d5 = 0.08535  # wrist 2 to end-effector
        self.d6 = 0.0921   # wrist 3 to end-effector
        
        # Joint limits (rad)
        self.left_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - fixed
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - fixed
            (-3.1417, -3.1415)  # wrist_3_joint (q6) - fixed
        ]
        self.right_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - fixed
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - fixed
            (3.1415, 3.1417)    # wrist_3_joint (q6) - fixed
        ]
        
        # Current joint states (for continuity)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e arm DH parameters (simplified model)
        self.d1 = 0.15185  # base height
        self.a2 = 0.24355  # upper arm length
        self.a3 = 0.21325  # forearm length
        self.d4 = 0.13105  # wrist 1 to end-effector
        self.d5 = 0.08535  # wrist 2 to end-effector
        self.d6 = 0.0921   # wrist 3 to end-effector
        
        # Joint limits (rad)
        self.left_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - fixed
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - fixed
            (-3.1417, -3.1415)  # wrist_3_joint (q6) - fixed
        ]
        self.right_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - fixed
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - fixed
            (3.1415, 3.1417)    # wrist_3_joint (q6) - fixed
        ]
        
        # Current joint states (for continuity)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e arm DH parameters (simplified model)
        self.d1 = 0.15185  # base height
        self.a2 = 0.24355  # upper arm length
        self.a3 = 0.21325  # forearm length
        self.d4 = 0.13105  # wrist 1 to end-effector
        self.d5 = 0.08535  # wrist 2 to end-effector
        self.d6 = 0.0921   # wrist 3 to end-effector
        
        # Joint limits (rad)
        self.joint_limits = [
            (-0.1, 0.1),              # shoulder_pan_joint (q1) - fixed
            (-np.pi/2, np.pi/2),      # shoulder_lift_joint (q2)
            (0, np.pi),               # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),      # wrist_1_joint (q4)
            (-0.1, 0.1),              # wrist_2_joint (q5) - fixed
            (-3.15, -3.13) if 'left' in self.get_name() else (3.13, 3.15) # wrist_3_joint (q6) - fixed
        ]
        
        # Current joint states (for continuity)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e arm DH parameters (simplified model)
        self.d1 = 0.15185  # base height
        self.a2 = 0.24355  # upper arm length
        self.a3 = 0.21325  # forearm length
        self.d4 = 0.13105  # wrist 1 to end-effector
        self.d5 = 0.08535  # wrist 2 to end-effector
        self.d6 = 0.0921   # wrist 3 to end-effector
        
        # Joint limits (rad)
        self.joint_limits = [
            (-2*np.pi, 2*np.pi),  # base
            (-np.pi, np.pi),       # shoulder
            (-np.pi, np.pi),       # elbow
            (-2*np.pi, 2*np.pi),  # wrist_1
            (-2*np.pi, 2*np.pi),  # wrist_2
            (-2*np.pi, 2*np.pi),  # wrist_3
        ]
        
        # Current joint states (for continuity)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # Actual joint states (read from /joint_states)
        self.left_actual_joints = None
        self.right_actual_joints = None
        
        # Target joint states (commands to be published)
        self.left_target_joints = None
        self.right_target_joints = None

        # Most recent gripper command value (0.0=open, 1.0=close), used for logging
        self.left_gripper_received = 0.0
        self.right_gripper_received = 0.0
        
        # Joint smoothing coefficient (reduce jitter)
        self.joint_smoothing = 0.2  # lower hysteresis, improve following
        
        # Debug counter
        self.debug_counter = 0
        
        # Previous target position (for change detection)
        self.left_last_target = None
        self.right_last_target = None
        # 5 mm change threshold (can be overridden via parameter)
        # self.position_change_threshold already set from parameters
        
        # Create timer to periodically publish complete joint state (for RViz/robot_state_publisher)
        period = 1.0 / self.joint_state_rate if self.joint_state_rate > 0.0 else 1.0
        self.joint_state_timer = self.create_timer(period, self.publish_complete_joint_state)

        # Publish once immediately to avoid missing TFs during RViz startup
        self.publish_complete_joint_state()
        
        # Base positions and rotation (from URDF)
        # Both arms rotated -90° (-1.5708 rad), i.e., counterclockwise
        self.left_base_pos = np.array([0.5, 0.5, 0.775])
        self.right_base_pos = np.array([-0.5, 0.5, 0.775])
        self.base_rotation = -np.pi/2  # -90° rotation
        
        # Planar IK workspace bounds (simplified geometry to avoid unreachable targets)
        # Expose margin via parameter for on-site tuning
        self.declare_parameter('workspace_margin', 0.02)
        margin = float(self.get_parameter('workspace_margin').value)
        self.x_min = 0.05
        self.x_max = self.a2 + self.a3 - max(0.01, margin)
        self.z_min = self.d1 - max(0.01, margin)
        self.z_max = self.d1 + self.a2 + self.a3 - max(0.01, margin)

        # Axis direction and gain adjustments:
        # - invert_relative_z: invert teleop ΔZ (enable if world "up" is negative)
        # - relative_z_gain: gain factor for relative ΔZ to boost upward motion
        self.declare_parameter('invert_relative_z', True)
        self.declare_parameter('relative_z_gain', 2.0)
        self.invert_relative_z = bool(self.get_parameter('invert_relative_z').value)
        self.relative_z_gain = float(self.get_parameter('relative_z_gain').value)

        # Compute planar FK from current joints as initial absolute target
        def _fk_xzyaw(j):
            q1, q2, q3, q4, q5, q6 = j
            x = (self.a2 * np.cos(q2) + self.a3 * np.cos(q2 + q3))
            z = self.d1 + self.a2 * np.sin(q2) + self.a3 * np.sin(q2 + q3)
            yaw = q4
            return float(x), float(z), float(yaw)

        lx, lz, lyaw = _fk_xzyaw(self.left_current_joints)
        rx, rz, ryaw = _fk_xzyaw(self.right_current_joints)

        # Absolute end-effector goal (integrated from relative increments)
        self.left_goal = {
            'x': np.clip(lx, self.x_min, self.x_max),
            'z': np.clip(lz, self.z_min, self.z_max),
            'yaw': lyaw,
            'last_rel': np.array([0.0, 0.0, 0.0], dtype=float)  # Last received [ΔX, ΔZ, ΔYaw]
        }
        self.right_goal = {
            'x': np.clip(rx, self.x_min, self.x_max),
            'z': np.clip(rz, self.z_min, self.z_max),
            'yaw': ryaw,
            'last_rel': np.array([0.0, 0.0, 0.0], dtype=float)
        }

        self.get_logger().info('End-effector command converter node started')

        # Runtime safety state
        self.state_enabled = True   # allow motion (controlled via SetBool)
        self.estop_active = False   # E-Stop active (Trigger to activate, reset to clear)

        # Reset reference/pose: service and topic
        self.reset_srv = self.create_service(Trigger, '/teleop/reset_reference', self.on_reset_reference)
        self.reset_sub = self.create_subscription(Empty, '/teleop/reset', self.on_reset_reference_msg, 10)

        # Enable/disable and E-Stop services
        self.enable_srv = self.create_service(SetBool, '/teleop/enable', self.on_enable)
        self.estop_srv = self.create_service(Trigger, '/teleop/estop', self.on_estop)
    
    def world_to_base_frame(self, world_x, world_y, world_z, base_pos):
        """
        Convert target from world frame to robot base frame.
        The arm base is rotated -90° (counterclockwise), requiring coordinate transform.
        
        World frame: X forward, Y left
        Base frame (after -90° rotation): X left, Y backward
        """
        # Position relative to base
        rel_x = world_x - base_pos[0]
        rel_y = world_y - base_pos[1]
        rel_z = world_z - base_pos[2]
        
        # Rotation transform: world → base after -90° rotation
        # Rotation matrix R(-90°):
        # [cos(-90°)  -sin(-90°)]   [0   1]
        # [sin(-90°)   cos(-90°)] = [-1  0]
        # 
        # new = R * old
        base_x = rel_y   # World Y → Base X (left becomes forward)
        base_y = -rel_x  # World X → Base -Y (forward becomes right, hence negative)
        base_z = rel_z
        
        return base_x, base_y, base_z

    def reset_internal_state(self):
        """
        Reset model initial position/reference:
        - Set current joint states of both arms to default values
        - Reset absolute end-effector goals and relative increments
        - Publish full joint state immediately for RViz sync
        """
        # Clear E-Stop; do not auto-enable — requires explicit enable
        self.estop_active = False
        # Initial joint values
        left_init = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -np.pi])
        right_init = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, np.pi])

        self.left_current_joints = left_init.copy()
        self.right_current_joints = right_init.copy()

        # Compute X-Z-Yaw from initial joints as new absolute targets
        def _fk_xzyaw(j):
            q1, q2, q3, q4, q5, q6 = j
            x = (self.a2 * np.cos(q2) + self.a3 * np.cos(q2 + q3))
            z = self.d1 + self.a2 * np.sin(q2) + self.a3 * np.sin(q2 + q3)
            yaw = q4
            return float(x), float(z), float(yaw)

        lx, lz, lyaw = _fk_xzyaw(self.left_current_joints)
        rx, rz, ryaw = _fk_xzyaw(self.right_current_joints)

        self.left_goal = {
            'x': np.clip(lx, self.x_min, self.x_max),
            'z': np.clip(lz, self.z_min, self.z_max),
            'yaw': lyaw,
            'last_rel': np.array([0.0, 0.0, 0.0], dtype=float)
        }
        self.right_goal = {
            'x': np.clip(rx, self.x_min, self.x_max),
            'z': np.clip(rz, self.z_min, self.z_max),
            'yaw': ryaw,
            'last_rel': np.array([0.0, 0.0, 0.0], dtype=float)
        }

        # Update all joint positions dictionary
        self.all_joint_positions.update({
            'left_shoulder_pan_joint': float(self.left_current_joints[0]),
            'left_shoulder_lift_joint': float(self.left_current_joints[1]),
            'left_elbow_joint': float(self.left_current_joints[2]),
            'left_wrist_1_joint': float(self.left_current_joints[3]),
            'left_wrist_2_joint': float(self.left_current_joints[4]),
            'left_wrist_3_joint': float(self.left_current_joints[5]),
            'right_shoulder_pan_joint': float(self.right_current_joints[0]),
            'right_shoulder_lift_joint': float(self.right_current_joints[1]),
            'right_elbow_joint': float(self.right_current_joints[2]),
            'right_wrist_1_joint': float(self.right_current_joints[3]),
            'right_wrist_2_joint': float(self.right_current_joints[4]),
            'right_wrist_3_joint': float(self.right_current_joints[5]),
        })

        # Immediately publish one joint state to reset RViz model to initial pose
        self.publish_complete_joint_state()

    def on_reset_reference(self, request, response):
        try:
            self.reset_internal_state()
            response.success = True
            response.message = 'Teleop reference and model initial pose reset'
        except Exception as e:
            response.success = False
            response.message = f'Reset failed: {e}'
        self.get_logger().info(response.message)
        return response

    def on_reset_reference_msg(self, _msg: Empty):
        # Trigger same logic via topic as the service
        try:
            self.reset_internal_state()
            self.get_logger().info('Received /teleop/reset, Teleop reference and model initial pose reset')
        except Exception as e:
            self.get_logger().error(f'/teleop/reset reset failed: {e}')
    
    # Safety control: enable/disable and emergency stop
    def on_enable(self, request: SetBool.Request, response: SetBool.Response):
        try:
            if bool(request.data):
                if self.estop_active:
                    response.success = False
                    response.message = 'Emergency stop active; call /teleop/reset or clear e-stop first'
                else:
                    self.state_enabled = True
                    response.success = True
                    response.message = 'Teleop enabled'
            else:
                self.state_enabled = False
                response.success = True
                response.message = 'Teleop disabled'
        except Exception as e:
            response.success = False
            response.message = f'Enable/disable handling failed: {e}'
        self.get_logger().info(response.message)
        return response

    def on_estop(self, request, response):
        try:
            self.estop_active = True
            self.state_enabled = False
            response.success = True
            response.message = 'Emergency stop engaged: Teleop output disabled'
        except Exception as e:
            response.success = False
            response.message = f'E-stop handling failed: {e}'
        self.get_logger().error(response.message)
        return response

    def _check_motion_allowed(self) -> bool:
        return (not self.estop_active) and self.state_enabled
    
    def simple_ik_planar(self, target_x, target_y, target_z, target_rot_z, current_joints, arm='left', interpret_relative=True):
        """
        Simplified IK (for XZ plane position + yaw)
        
        Input mapping (from vision/gesture 3DOF):
        - Wrist X → end-effector X
        - Wrist Y (height) → end-effector Z
        - Wrist rotation angle → end-effector yaw (via wrist_1), with light noise suppression
        
        Joint usage strategy:
        - Shoulder pan (q1) fixed at 0 (face forward)
        - Shoulder lift (q2) + elbow (q3) → control XZ position
        - Wrist_1 (q4) → controls yaw
        - Wrist_2 (q5) and wrist_3 (q6) fixed (0 and ±pi respectively), set per arm side
        """

        # Target interpretation: if relative displacement, map X→local_x, Y→local_z directly; otherwise do world→base transform
        if interpret_relative:
            local_x = target_x   # Absolute end-effector X (already handled externally)
            local_z = target_y   # Absolute end-effector Z (already handled externally)
        else:
            base_pos = self.left_base_pos if arm == 'left' else self.right_base_pos
            local_x, _local_y, local_z = self.world_to_base_frame(target_x, target_y, target_z, base_pos)

        # Apply simple low-pass filter to yaw to suppress jitter (use previous frame if present)
        if not hasattr(self, 'last_yaw'):
            self.last_yaw = { 'left': 0.0, 'right': 0.0 }
        alpha = 0.3  # New value weight (0.3), old value (0.7)
        target_rot_z_filtered = alpha * target_rot_z + (1 - alpha) * self.last_yaw[arm]
        self.last_yaw[arm] = target_rot_z_filtered

        def forward_kinematics_simplified(joints):
            """Simplified forward kinematics (compute local XZ position and yaw)"""
            q1, q2, q3, q4, q5, q6 = joints

            # When q1≈0, end-effector local y≈0; primarily controlling x-z plane
            x = (self.a2 * np.cos(q2) + self.a3 * np.cos(q2 + q3))
            z = self.d1 + self.a2 * np.sin(q2) + self.a3 * np.sin(q2 + q3)

            # Yaw provided only by wrist_1 to avoid rotation noise from wrist_2/3
            rot_z = q4
            return np.array([x, z, rot_z])

        def cost_function(joints):
            fk_x, fk_z, fk_yaw = forward_kinematics_simplified(joints)

            # Position error: consider x and z only; ignore y to match input definition
            pos_error = (fk_x - local_x) ** 2 + 1.2 * (fk_z - local_z) ** 2

            # Yaw error (using filtered target angle)
            rot_error = (fk_yaw - target_rot_z_filtered) ** 2

            # Smoothness penalty: limit jump relative to previous state (reinforce elbow involvement)
            q1c, q2c, q3c, q4c, q5c, q6c = current_joints
            q1n, q2n, q3n, q4n, q5n, q6n = joints
            joint_change = (
                (q1n - q1c) ** 2 +
                0.8 * (q2n - q2c) ** 2 +
                0.3 * (q3n - q3c) ** 2 +
                (q4n - q4c) ** 2 +
                (q5n - q5c) ** 2 +
                (q6n - q6c) ** 2
            ) * 0.25

            # Fixed-joint penalty: keep q1≈0, q5≈0, q6 near side-specific targets
            q1, q2, q3, q4, q5, q6 = joints
            fixed_penalty = (q1 - 0.0) ** 2 * 5.0 + (q5 - 0.0) ** 2 * 5.0
            target_q6 = -np.pi if arm == 'left' else np.pi
            fixed_penalty += (q6 - target_q6) ** 2 * 5.0

            # Posture penalty: encourage elbow (q3) near 40°, shoulder (q2) near 0°
            q3_pref = np.deg2rad(40.0)
            posture_penalty = 0.12 * (q3 - q3_pref) ** 2 + 0.02 * q2 ** 2

            return pos_error + rot_error * 0.8 + joint_change + fixed_penalty + posture_penalty

        # Perform optimization using SLSQP
        result = minimize(
            cost_function,
            current_joints,
            method='SLSQP',
            bounds=self.joint_limits,
            options={'maxiter': 100, 'ftol': 1e-3}
        )

        if result.success:
            # Smooth the result overall to further suppress noise
            smoothed = self.joint_smoothing * result.x + (1 - self.joint_smoothing) * current_joints
            # Clamp fixed joints near targets to avoid numerical drift
            smoothed[0] = 0.0  # q1 fixed
            smoothed[4] = 0.0  # q5 fixed
            smoothed[5] = -np.pi if arm == 'left' else np.pi  # q6 fixed
            return smoothed
        else:
            self.get_logger().warn('IK failed, keeping current position')
            return current_joints
    
    def create_joint_trajectory_msg(self, joint_positions, joint_names, arm='left'):
        """Create joint trajectory message"""
        msg = JointTrajectory()
        msg.joint_names = joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 0.1 s
        
        msg.points = [point]
        
        return msg
    
    def publish_joint_state(self, joint_positions, joint_names):
        """Publish specified joint state for RViz visualization"""
        # Update positions for corresponding joints
        for i, name in enumerate(joint_names):
            if name in self.all_joint_positions:
                self.all_joint_positions[name] = joint_positions[i]
        
        # Publish only current updated joint states (including gripper joints)
        # Determine left or right arm
        if 'left' in joint_names[0]:
            arm = 'left'
        else:
            arm = 'right'
        
        # Build list of joints to publish (only current arm joints)
        arm_joint_names = []
        arm_joint_positions = []
        
        for name, position in self.all_joint_positions.items():
            if arm in name:
                arm_joint_names.append(name)
                arm_joint_positions.append(position)
        
        # Publish current arm joint state
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = arm_joint_names
        msg.position = arm_joint_positions
        msg.velocity = []
        msg.effort = []
        
        self.joint_state_pub.publish(msg)
    
    def publish_complete_joint_state(self):
        """Periodically publish full joint state (low rate, 1 Hz)"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.all_joint_positions.keys())
        msg.position = list(self.all_joint_positions.values())
        msg.velocity = []
        msg.effort = []
        
        self.joint_state_pub.publish(msg)
    
    def update_debug_counter(self):
        """Update debug counters; periodically output comparison"""
        self.debug_counter += 1
        if self.debug_counter % 30 == 0:
            self.compare_joint_states()
    
    def compare_joint_states(self):
        """Display current joint state debug info"""
        self.get_logger().info('='*80)
        self.get_logger().info('Current joint state (expected values in RViz simulation mode)')

        
        # Left arm state
        if self.left_target_joints is not None:
            self.get_logger().info(f'  Left arm joint state:')
            self.get_logger().info(f'  Joint angles: [{" ".join([f"{j:7.3f}" for j in self.left_target_joints])}]')
            self.get_logger().info(f'  Angles (deg): [{" ".join([f"{np.degrees(j):7.1f}" for j in self.left_target_joints])}]')
            # Left gripper (actual: inferred from joint value; semantics match input 0=open,1=close)
            try:
                _lk = self.all_joint_positions.get('left_robotiq_85_left_knuckle_joint', None)
                if _lk is not None:
                    _actual_left = 1.0 - np.clip(_lk / 0.8, 0.0, 1.0)
                else:
                    _actual_left = float(self.left_gripper_received)
            except Exception:
                _actual_left = float(self.left_gripper_received)
            self.get_logger().info(f'  Left gripper actual state: {_actual_left:.2f}')
        
        # Right arm state
        if self.right_target_joints is not None:
            self.get_logger().info(f'  Right arm joint state:')
            self.get_logger().info(f'  Joint angles: [{" ".join([f"{j:7.3f}" for j in self.right_target_joints])}]')
            self.get_logger().info(f'  Angles (deg): [{" ".join([f"{np.degrees(j):7.1f}" for j in self.right_target_joints])}]')
            # Right gripper (actual: inferred from joint value; semantics match input 0=open,1=close)
            try:
                _rk = self.all_joint_positions.get('right_robotiq_85_left_knuckle_joint', None)
                if _rk is not None:
                    _actual_right = 1.0 - np.clip(_rk / 0.8, 0.0, 1.0)
                else:
                    _actual_right = float(self.right_gripper_received)
            except Exception:
                _actual_right = float(self.right_gripper_received)
            self.get_logger().info(f'  Right gripper actual state: {_actual_right:.2f}')
        
        self.get_logger().info('='*80 + '\n')
    
    def left_ee_callback(self, msg):
        """Process left arm end-effector relative target"""
        if len(msg.data) < 3:
            self.get_logger().warn('Left arm EE target message has insufficient data')
            return
        # Safety guard: ignore if not enabled or emergency stop active
        if not self._check_motion_allowed():
            return
        
        rel_dx, rel_dz, rel_dyaw = [float(v) for v in msg.data[:3]]
        # Calculate increment from last relative values to avoid drift from repeated integration
        last_rel = self.left_goal['last_rel']
        inc = np.array([rel_dx, rel_dz, rel_dyaw], dtype=float) - last_rel
        # Small threshold filtering (separately for translation and rotation) 
        pos_inc_norm = np.hypot(inc[0], inc[1])
        if pos_inc_norm < 0.003 and abs(inc[2]) < np.deg2rad(1.0):
            return
        # Record as new last relative values
        self.left_goal['last_rel'] = np.array([rel_dx, rel_dz, rel_dyaw], dtype=float)
        
        # Integrate to absolute end-effector target and clip
        self.left_goal['x'] = float(np.clip(self.left_goal['x'] + inc[0], self.x_min, self.x_max))
        dz_raw = -inc[1] if self.invert_relative_z else inc[1]
        dz_inc = self.relative_z_gain * dz_raw
        self.left_goal['z'] = float(np.clip(self.left_goal['z'] + dz_inc, self.z_min, self.z_max))
        yaw_new = self.left_goal['yaw'] + inc[2]
        # Wrap to [-pi, pi]
        yaw_new = (yaw_new + np.pi) % (2*np.pi) - np.pi
        self.left_goal['yaw'] = float(yaw_new)
        
        # Debug info
        if not hasattr(self, '_left_callback_count'):
            self._left_callback_count = 0
        self._left_callback_count += 1
        
        # Optionally parse seq (4th element)
        seq = None
        if len(msg.data) >= 4:
            try:
                seq = int(msg.data[3])
            except Exception:
                seq = None
        # Debug: show relative displacement and yaw by seq or internal counter
        if (seq is not None and seq % self.debug_log_every_n == 0) or \
           (seq is None and self._left_callback_count % self.debug_log_every_n == 1):
            suffix = f" (seq={seq})" if seq is not None else ""
            self.get_logger().info(
                f'[Left arm] Received relative target: ΔX={rel_dx:.3f}m, ΔZ={rel_dz:.3f}m, ΔYaw={rel_dyaw:.3f}rad{suffix}'
            )
            # After receiving relative target, output left gripper (received)
            self.get_logger().info(f'Left gripper received target: {self.left_gripper_received:.2f}')
        
        # Solve inverse kinematics
        try:
            joint_positions = self.simple_ik_planar(
                self.left_goal['x'], self.left_goal['z'], 0.0, self.left_goal['yaw'],
                self.left_current_joints, 'left', interpret_relative=True
            )
            
            # Save previous joint state to compare changes
            self.left_actual_joints = self.left_current_joints.copy()
            
            # Update current state
            self.left_current_joints = joint_positions
            
            # Save target joint state
            self.left_target_joints = joint_positions.copy()
            
            # Update debug counter
            self.update_debug_counter()
            
            # Publish joint command
            joint_names = [
                'left_shoulder_pan_joint',
                'left_shoulder_lift_joint',
                'left_elbow_joint',
                'left_wrist_1_joint',
                'left_wrist_2_joint',
                'left_wrist_3_joint'
            ]
            
            trajectory_msg = self.create_joint_trajectory_msg(joint_positions, joint_names, 'left')
            self.left_joint_pub.publish(trajectory_msg)
            
            # Debug output: published joint angles
            if (seq is not None and seq % self.debug_log_every_n == 0) or \
               (seq is None and self._left_callback_count % self.debug_log_every_n == 1):
                self.get_logger().info(f'[Left arm] Published joint command: [{", ".join([f"{j:.3f}" for j in joint_positions])}]')
                joint_degrees = [f"{np.degrees(j):6.1f}°" for j in joint_positions]
                self.get_logger().info(f'         Angles: [{", ".join(joint_degrees)}]')
                self.get_logger().info(f'         Target EE (x,z,yaw): ({self.left_goal["x"]:.3f}, {self.left_goal["z"]:.3f}, {self.left_goal["yaw"]:.3f})')
            
            # Publish joint state for RViz
            self.publish_joint_state(joint_positions, joint_names)
        except Exception as e:
            self.get_logger().error(f'Left arm IK solve failed: {e}')
    
    def left_gripper_callback(self, msg):
        """Process left gripper command"""
        if len(msg.data) < 1:
            return
        if not self._check_motion_allowed():
            return
        
        gripper_position = msg.data[0]  # 0.0=open, 1.0=close
        # Record most recently received command value for logging
        self.left_gripper_received = float(gripper_position)
        
        # Update gripper joint state
        # Robotiq 85 gripper: 0.0=open, 0.8=fully closed
        gripper_value = (1.0 - gripper_position) * 0.8
        
        self.all_joint_positions['left_robotiq_85_left_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_right_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_left_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_right_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_left_finger_tip_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_right_finger_tip_joint'] = gripper_value
        
        # Log output placed elsewhere per user request
    
    def right_gripper_callback(self, msg):
        """Process right gripper command"""
        if len(msg.data) < 1:
            return
        if not self._check_motion_allowed():
            return
        
        gripper_position = msg.data[0]  # 0.0=open, 1.0=close (inverted to match expectation)
        # Record most recently received command value
        self.right_gripper_received = float(gripper_position)
        gripper_value = (1.0 - gripper_position) * 0.8
        
        self.all_joint_positions['right_robotiq_85_left_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_right_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_left_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_right_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_left_finger_tip_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_right_finger_tip_joint'] = gripper_value
        
        # Log output placed elsewhere per user request
    
    def right_ee_callback(self, msg):
        """Process right arm end-effector target"""
        if len(msg.data) < 3:
            return
        if not self._check_motion_allowed():
            return
        
        rel_dx, rel_dz, rel_dyaw = [float(v) for v in msg.data[:3]]
        last_rel = self.right_goal['last_rel']
        inc = np.array([rel_dx, rel_dz, rel_dyaw], dtype=float) - last_rel
        pos_inc_norm = np.hypot(inc[0], inc[1])
        if pos_inc_norm < 0.003 and abs(inc[2]) < np.deg2rad(1.0):
            return
        self.right_goal['last_rel'] = np.array([rel_dx, rel_dz, rel_dyaw], dtype=float)
        
        self.right_goal['x'] = float(np.clip(self.right_goal['x'] + inc[0], self.x_min, self.x_max))
        dz_raw_r = -inc[1] if self.invert_relative_z else inc[1]
        dz_inc_r = self.relative_z_gain * dz_raw_r
        self.right_goal['z'] = float(np.clip(self.right_goal['z'] + dz_inc_r, self.z_min, self.z_max))
        yaw_new = self.right_goal['yaw'] + inc[2]
        yaw_new = (yaw_new + np.pi) % (2*np.pi) - np.pi
        self.right_goal['yaw'] = float(yaw_new)
        
        # Debug info
        if not hasattr(self, '_right_callback_count'):
            self._right_callback_count = 0
        self._right_callback_count += 1
        
        # Optionally parse seq (4th element)
        seq = None
        if len(msg.data) >= 4:
            try:
                seq = int(msg.data[3])
            except Exception:
                seq = None
        # Debug: display relative displacement and yaw by seq or internal count
        if (seq is not None and seq % self.debug_log_every_n == 0) or \
           (seq is None and self._right_callback_count % self.debug_log_every_n == 1):
            suffix = f" (seq={seq})" if seq is not None else ""
            self.get_logger().info(f'[Right arm] Received relative target: ΔX={rel_dx:.3f}m, ΔZ={rel_dz:.3f}m, ΔYaw={rel_dyaw:.3f}rad{suffix}')
            # After receiving relative target, output right gripper (received)
            self.get_logger().info(f'Right gripper received state: {self.right_gripper_received:.2f}')
        
        # Solve inverse kinematics
        try:
            joint_positions = self.simple_ik_planar(
                self.right_goal['x'], self.right_goal['z'], 0.0, self.right_goal['yaw'],
                self.right_current_joints, 'right', interpret_relative=True
            )
            
            # Save previous joint states for comparison
            self.right_actual_joints = self.right_current_joints.copy()
            
            # Update current joint states
            self.right_current_joints = joint_positions
            
            # Save target joint states
            self.right_target_joints = joint_positions.copy()
            
            # Publish joint command
            joint_names = [
                'right_shoulder_pan_joint',
                'right_shoulder_lift_joint',
                'right_elbow_joint',
                'right_wrist_1_joint',
                'right_wrist_2_joint',
                'right_wrist_3_joint'
            ]
            
            trajectory_msg = self.create_joint_trajectory_msg(joint_positions, joint_names, 'right')
            self.right_joint_pub.publish(trajectory_msg)
            
            # Debug output: published joint angles
            if (seq is not None and seq % self.debug_log_every_n == 0) or \
               (seq is None and self._right_callback_count % self.debug_log_every_n == 1):
                self.get_logger().info(f'[Right arm] Published joint command: [{", ".join([f"{j:.3f}" for j in joint_positions])}]')
                joint_degrees = [f"{np.degrees(j):6.1f}°" for j in joint_positions]
                self.get_logger().info(f'         Angles: [{", ".join(joint_degrees)}]')
                self.get_logger().info(f'         Target EE (x,z,yaw): ({self.right_goal["x"]:.3f}, {self.right_goal["z"]:.3f}, {self.right_goal["yaw"]:.3f})')
            
            # Publish joint state for RViz
            self.publish_joint_state(joint_positions, joint_names)
        except Exception as e:
            self.get_logger().error(f'Right arm IK solve failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    node = EndEffectorCommander()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
