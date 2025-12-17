#!/usr/bin/env python3
"""
末端执行器命令转换节点
将末端执行器目标位置转换为关节命令
使用简化的逆运动学求解
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Header
from std_msgs.msg import Empty
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.optimize import minimize

class EndEffectorCommander(Node):
    def __init__(self):
        super().__init__('endeffector_commander')
        # 调试：使用序号对齐日志
        self.declare_parameter('debug_log_every_n', 30)
        self.declare_parameter('joint_state_rate', 10.0)  # Hz
        self.declare_parameter('position_change_threshold', 0.005)  # m
        self.debug_log_every_n = int(self.get_parameter('debug_log_every_n').value)
        self.joint_state_rate = float(self.get_parameter('joint_state_rate').value)
        self.position_change_threshold = float(self.get_parameter('position_change_threshold').value)
        
        # 订阅末端执行器目标
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
        
        # 订阅夹爪命令
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
        
        # 发布关节轨迹命令
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
        
        # 发布关节状态(用于RViz可视化)
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        # 跟踪所有关节的状态
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
            # Gripper joints (保持固定)
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
        
        # UR3e机械臂DH参数 (简化模型)
        self.d1 = 0.15185  # 基座高度
        self.a2 = 0.24355  # 大臂长度
        self.a3 = 0.21325  # 小臂长度
        self.d4 = 0.13105  # 手腕1到末端
        self.d5 = 0.08535  # 手腕2到末端
        self.d6 = 0.0921   # 手腕3到末端
        
        # 关节限制 (rad)
        self.left_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - 固定
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - 固定
            (-3.1417, -3.1415)  # wrist_3_joint (q6) - 固定
        ]
        self.right_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - 固定
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - 固定
            (3.1415, 3.1417)    # wrist_3_joint (q6) - 固定
        ]
        
        # 当前关节状态(用于连续性)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e机械臂DH参数 (简化模型)
        self.d1 = 0.15185  # 基座高度
        self.a2 = 0.24355  # 大臂长度
        self.a3 = 0.21325  # 小臂长度
        self.d4 = 0.13105  # 手腕1到末端
        self.d5 = 0.08535  # 手腕2到末端
        self.d6 = 0.0921   # 手腕3到末端
        
        # 关节限制 (rad)
        self.left_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - 固定
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - 固定
            (-3.1417, -3.1415)  # wrist_3_joint (q6) - 固定
        ]
        self.right_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - 固定
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - 固定
            (3.1415, 3.1417)    # wrist_3_joint (q6) - 固定
        ]
        
        # 当前关节状态(用于连续性)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e机械臂DH参数 (简化模型)
        self.d1 = 0.15185  # 基座高度
        self.a2 = 0.24355  # 大臂长度
        self.a3 = 0.21325  # 小臂长度
        self.d4 = 0.13105  # 手腕1到末端
        self.d5 = 0.08535  # 手腕2到末端
        self.d6 = 0.0921   # 手腕3到末端
        
        # 关节限制 (rad)
        self.left_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - 固定
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - 固定
            (-3.1417, -3.1415)  # wrist_3_joint (q6) - 固定
        ]
        self.right_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - 固定
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - 固定
            (3.1415, 3.1417)    # wrist_3_joint (q6) - 固定
        ]
        
        # 当前关节状态(用于连续性)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e机械臂DH参数 (简化模型)
        self.d1 = 0.15185  # 基座高度
        self.a2 = 0.24355  # 大臂长度
        self.a3 = 0.21325  # 小臂长度
        self.d4 = 0.13105  # 手腕1到末端
        self.d5 = 0.08535  # 手腕2到末端
        self.d6 = 0.0921   # 手腕3到末端
        
        # 关节限制 (rad)
        self.left_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - 固定
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - 固定
            (-3.1417, -3.1415)  # wrist_3_joint (q6) - 固定
        ]
        self.right_joint_limits = [
            (-0.01, 0.01),      # shoulder_pan_joint (q1) - 固定
            (-np.pi, 0),        # shoulder_lift_joint (q2)
            (0, 2*np.pi/3),     # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),# wrist_1_joint (q4)
            (-0.01, 0.01),      # wrist_2_joint (q5) - 固定
            (3.1415, 3.1417)    # wrist_3_joint (q6) - 固定
        ]
        
        # 当前关节状态(用于连续性)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e机械臂DH参数 (简化模型)
        self.d1 = 0.15185  # 基座高度
        self.a2 = 0.24355  # 大臂长度
        self.a3 = 0.21325  # 小臂长度
        self.d4 = 0.13105  # 手腕1到末端
        self.d5 = 0.08535  # 手腕2到末端
        self.d6 = 0.0921   # 手腕3到末端
        
        # 关节限制 (rad)
        self.joint_limits = [
            (-0.1, 0.1),              # shoulder_pan_joint (q1) - 固定
            (-np.pi/2, np.pi/2),      # shoulder_lift_joint (q2)
            (0, np.pi),               # elbow_joint (q3)
            (-2*np.pi, 2*np.pi),      # wrist_1_joint (q4)
            (-0.1, 0.1),              # wrist_2_joint (q5) - 固定
            (-3.15, -3.13) if 'left' in self.get_name() else (3.13, 3.15) # wrist_3_joint (q6) - 固定
        ]
        
        # 当前关节状态(用于连续性)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # UR3e机械臂DH参数 (简化模型)
        self.d1 = 0.15185  # 基座高度
        self.a2 = 0.24355  # 大臂长度
        self.a3 = 0.21325  # 小臂长度
        self.d4 = 0.13105  # 手腕1到末端
        self.d5 = 0.08535  # 手腕2到末端
        self.d6 = 0.0921   # 手腕3到末端
        
        # 关节限制 (rad)
        self.joint_limits = [
            (-2*np.pi, 2*np.pi),  # 基座
            (-np.pi, np.pi),       # 肩部
            (-np.pi, np.pi),       # 肘部
            (-2*np.pi, 2*np.pi),  # 腕1
            (-2*np.pi, 2*np.pi),  # 腕2
            (-2*np.pi, 2*np.pi),  # 腕3
        ]
        
        # 当前关节状态(用于连续性)
        self.left_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -3.1416])
        self.right_current_joints = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416])
        
        # 实际关节状态（从/joint_states读取）
        self.left_actual_joints = None
        self.right_actual_joints = None
        
        # 目标关节状态（发布的命令）
        self.left_target_joints = None
        self.right_target_joints = None

        # 夹爪最近一次接收的命令值（0.0=打开, 1.0=关闭），用于在指定日志位置展示
        self.left_gripper_received = 0.0
        self.right_gripper_received = 0.0
        
        # 关节平滑系数（减少抖动）
        self.joint_smoothing = 0.2  # 降低迟滞，提高跟随
        
        # 调试计数器
        self.debug_counter = 0
        
        # 上一次的目标位置（用于检测变化）
        self.left_last_target = None
        self.right_last_target = None
        # 5mm的变化阈值（可通过参数覆盖）
        # self.position_change_threshold 已在参数读取处设置
        
        # 创建定时器定期发布完整的关节状态（用于RViz/robot_state_publisher）
        period = 1.0 / self.joint_state_rate if self.joint_state_rate > 0.0 else 1.0
        self.joint_state_timer = self.create_timer(period, self.publish_complete_joint_state)

        # 立即发布一次完整关节状态，避免 RViz 初始阶段缺少 TF
        self.publish_complete_joint_state()
        
        # 基座位置和旋转（从URDF）
        # 两个机械臂都旋转了-90度（-1.5708 rad），即逆时针旋转90度
        self.left_base_pos = np.array([0.5, 0.5, 0.775])
        self.right_base_pos = np.array([-0.5, 0.5, 0.775])
        self.base_rotation = -np.pi/2  # -90度旋转
        
        # 平面IK工作空间夹限（基于简化几何，避免不可达目标）
        # 通过参数暴露边界余量以便现场调节
        self.declare_parameter('workspace_margin', 0.02)
        margin = float(self.get_parameter('workspace_margin').value)
        self.x_min = 0.05
        self.x_max = self.a2 + self.a3 - max(0.01, margin)
        self.z_min = self.d1 - max(0.01, margin)
        self.z_max = self.d1 + self.a2 + self.a3 - max(0.01, margin)

        # 坐标轴方向与增益修正：
        # - invert_relative_z: 是否反转 Teleop 的 ΔZ（世界坐标“上”为负时开启）
        # - relative_z_gain: 相对ΔZ的增益因子，用于增强向上的幅度
        self.declare_parameter('invert_relative_z', True)
        self.declare_parameter('relative_z_gain', 2.0)
        self.invert_relative_z = bool(self.get_parameter('invert_relative_z').value)
        self.relative_z_gain = float(self.get_parameter('relative_z_gain').value)

        # 以当前关节状态计算平面FK，用作初始绝对目标
        def _fk_xzyaw(j):
            q1, q2, q3, q4, q5, q6 = j
            x = (self.a2 * np.cos(q2) + self.a3 * np.cos(q2 + q3))
            z = self.d1 + self.a2 * np.sin(q2) + self.a3 * np.sin(q2 + q3)
            yaw = q4
            return float(x), float(z), float(yaw)

        lx, lz, lyaw = _fk_xzyaw(self.left_current_joints)
        rx, rz, ryaw = _fk_xzyaw(self.right_current_joints)

        # 绝对末端目标（由相对量增量积分而来）
        self.left_goal = {
            'x': np.clip(lx, self.x_min, self.x_max),
            'z': np.clip(lz, self.z_min, self.z_max),
            'yaw': lyaw,
            'last_rel': np.array([0.0, 0.0, 0.0], dtype=float)  # 上次收到的[ΔX,ΔZ,ΔYaw]
        }
        self.right_goal = {
            'x': np.clip(rx, self.x_min, self.x_max),
            'z': np.clip(rz, self.z_min, self.z_max),
            'yaw': ryaw,
            'last_rel': np.array([0.0, 0.0, 0.0], dtype=float)
        }

        self.get_logger().info('末端执行器命令转换节点已启动')

        # 重置参考/姿态：服务与话题
        self.reset_srv = self.create_service(Trigger, '/teleop/reset_reference', self.on_reset_reference)
        self.reset_sub = self.create_subscription(Empty, '/teleop/reset', self.on_reset_reference_msg, 10)
    
    def world_to_base_frame(self, world_x, world_y, world_z, base_pos):
        """
        将世界坐标系的目标转换为机械臂基座坐标系
        机械臂基座旋转了-90度（逆时针），需要进行坐标转换
        
        世界坐标系：X向前，Y向左
        机械臂基座坐标系（旋转-90度后）：X向左，Y向后
        """
        # 相对于基座的位置
        rel_x = world_x - base_pos[0]
        rel_y = world_y - base_pos[1]
        rel_z = world_z - base_pos[2]
        
        # 旋转变换：将世界坐标转到旋转-90度后的基座坐标
        # 旋转矩阵 R(-90°):
        # [cos(-90°)  -sin(-90°)]   [0   1]
        # [sin(-90°)   cos(-90°)] = [-1  0]
        # 
        # 新坐标 = R * 旧坐标
        base_x = rel_y   # 世界的Y → 基座的X（向左变成向前）
        base_y = -rel_x  # 世界的X → 基座的-Y（向前变成向右，所以取负）
        base_z = rel_z
        
        return base_x, base_y, base_z

    def reset_internal_state(self):
        """
        重置模型初始位置/参考：
        - 将左右臂当前关节状态设为初始默认值
        - 重置左右末端绝对目标与相对增量
        - 立即发布一次完整关节状态，便于 RViz 同步
        """
        # 关节初始值
        left_init = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, -np.pi])
        right_init = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, np.pi])

        self.left_current_joints = left_init.copy()
        self.right_current_joints = right_init.copy()

        # 以初始关节计算对应的 X-Z-Yaw 作为新的绝对目标
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

        # 更新 all_joint_positions（含夹爪保持不变）
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

        # 立即发布一次关节状态，使 RViz 模型回到初始位姿
        self.publish_complete_joint_state()

    def on_reset_reference(self, request, response):
        try:
            self.reset_internal_state()
            response.success = True
            response.message = '已重置 Teleop 参考与模型初始位置'
        except Exception as e:
            response.success = False
            response.message = f'重置失败: {e}'
        self.get_logger().info(response.message)
        return response

    def on_reset_reference_msg(self, _msg: Empty):
        # 通过话题触发与服务同样的逻辑
        try:
            self.reset_internal_state()
            self.get_logger().info('收到 /teleop/reset，已重置 Teleop 参考与模型初始位置')
        except Exception as e:
            self.get_logger().error(f'/teleop/reset 重置失败: {e}')
    
    def simple_ik_planar(self, target_x, target_y, target_z, target_rot_z, current_joints, arm='left', interpret_relative=True):
        """
        简化的逆运动学求解(针对XZ平面位置 + 偏航角)
        
        输入映射(来自视觉/手势检测的3DOF)：
        - 手腕X位置 → 机械臂末端X位置
        - 手腕Y高度 → 机械臂末端Z高度
        - 手腕转动角 → 机械臂末端偏航角(腕1控制)，带轻量噪声抑制
        
        关节使用策略：
        - 肩pan(q1)固定为0(保持正对)
        - 肩lift(q2)+肘(q3) → 控制XZ位置
        - 腕1(q4) → 对应偏航角
        - 腕2(q5)、腕3(q6) 固定(分别为0和±pi)，按左右臂设定
        """

        # 目标解释：若为相对位移，则直接将 X→local_x，Y→local_z；否则做世界→基座变换
        if interpret_relative:
            local_x = target_x   # 目标末端绝对X（已在外部处理为绝对）
            local_z = target_y   # 目标末端绝对Z（已在外部处理为绝对）
        else:
            base_pos = self.left_base_pos if arm == 'left' else self.right_base_pos
            local_x, _local_y, local_z = self.world_to_base_frame(target_x, target_y, target_z, base_pos)

        # 对偏航角做简单的低通滤波，抑制抖动(上一帧存在则使用)
        if not hasattr(self, 'last_yaw'):
            self.last_yaw = { 'left': 0.0, 'right': 0.0 }
        alpha = 0.3  # 新值权重(0.3)，旧值(0.7)
        target_rot_z_filtered = alpha * target_rot_z + (1 - alpha) * self.last_yaw[arm]
        self.last_yaw[arm] = target_rot_z_filtered

        def forward_kinematics_simplified(joints):
            """简化正运动学(计算局部XZ位置与偏航角)"""
            q1, q2, q3, q4, q5, q6 = joints

            # q1≈0时，末端在局部y≈0，主要控制x-z平面
            x = (self.a2 * np.cos(q2) + self.a3 * np.cos(q2 + q3))
            z = self.d1 + self.a2 * np.sin(q2) + self.a3 * np.sin(q2 + q3)

            # 偏航仅由腕1提供，避免腕2/3引入旋转噪声
            rot_z = q4
            return np.array([x, z, rot_z])

        def cost_function(joints):
            fk_x, fk_z, fk_yaw = forward_kinematics_simplified(joints)

            # 位置误差：只考虑 x 与 z，y忽略以匹配输入定义
            pos_error = (fk_x - local_x) ** 2 + 1.2 * (fk_z - local_z) ** 2

            # 偏航误差(使用滤波后的目标角)
            rot_error = (fk_yaw - target_rot_z_filtered) ** 2

            # 平滑惩罚：限制相对上一状态的跳变（强化肘参与）
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

            # 固定关节保持惩罚：q1接近0，q5接近0，q6接近左右设定值
            q1, q2, q3, q4, q5, q6 = joints
            fixed_penalty = (q1 - 0.0) ** 2 * 5.0 + (q5 - 0.0) ** 2 * 5.0
            target_q6 = -np.pi if arm == 'left' else np.pi
            fixed_penalty += (q6 - target_q6) ** 2 * 5.0

            # 姿态偏好：更强地鼓励肘关节弯曲，避免全部由肩承担
            q3_pref = np.deg2rad(40.0)
            posture_penalty = 0.12 * (q3 - q3_pref) ** 2 + 0.02 * q2 ** 2

            return pos_error + rot_error * 0.8 + joint_change + fixed_penalty + posture_penalty

        # 使用优化求解，bounds沿用已定义的joint_limits
        result = minimize(
            cost_function,
            current_joints,
            method='SLSQP',
            bounds=self.joint_limits,
            options={'maxiter': 100, 'ftol': 1e-3}
        )

        if result.success:
            # 对结果进行整体平滑，进一步抑制噪声
            smoothed = self.joint_smoothing * result.x + (1 - self.joint_smoothing) * current_joints
            # 强制将固定关节钳制在目标附近，避免数值漂移
            smoothed[0] = 0.0  # q1 固定
            smoothed[4] = 0.0  # q5 固定
            smoothed[5] = -np.pi if arm == 'left' else np.pi  # q6 固定
            return smoothed
        else:
            self.get_logger().warn('逆运动学求解失败,保持当前位置')
            return current_joints
    
    def create_joint_trajectory_msg(self, joint_positions, joint_names, arm='left'):
        """创建关节轨迹消息"""
        msg = JointTrajectory()
        msg.joint_names = joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 0.1秒
        
        msg.points = [point]
        
        return msg
    
    def publish_joint_state(self, joint_positions, joint_names):
        """发布指定关节状态用于RViz可视化"""
        # 更新对应关节的位置
        for i, name in enumerate(joint_names):
            if name in self.all_joint_positions:
                self.all_joint_positions[name] = joint_positions[i]
        
        # 只发布当前更新的关节状态(包含夹爪关节)
        # 确定是左臂还是右臂
        if 'left' in joint_names[0]:
            arm = 'left'
        else:
            arm = 'right'
        
        # 构建要发布的关节列表(只包含当前手臂的关节)
        arm_joint_names = []
        arm_joint_positions = []
        
        for name, position in self.all_joint_positions.items():
            if arm in name:
                arm_joint_names.append(name)
                arm_joint_positions.append(position)
        
        # 发布当前手臂的关节状态
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = arm_joint_names
        msg.position = arm_joint_positions
        msg.velocity = []
        msg.effort = []
        
        self.joint_state_pub.publish(msg)
    
    def publish_complete_joint_state(self):
        """定期发布完整的关节状态（低频率，1Hz）"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.all_joint_positions.keys())
        msg.position = list(self.all_joint_positions.values())
        msg.velocity = []
        msg.effort = []
        
        self.joint_state_pub.publish(msg)
    
    def update_debug_counter(self):
        """更新调试计数器,定期输出对比结果"""
        self.debug_counter += 1
        if self.debug_counter % 30 == 0:
            self.compare_joint_states()
    
    def compare_joint_states(self):
        """显示当前关节状态调试信息"""
        self.get_logger().info('='*80)
        self.get_logger().info('当前关节状态 (RViz仿真模式下期望的实际值)')

        
        # 左臂状态
        if self.left_target_joints is not None:
            self.get_logger().info(f'  左臂关节状态:')
            self.get_logger().info(f'  关节角度: [{" ".join([f"{j:7.3f}" for j in self.left_target_joints])}]')
            self.get_logger().info(f'  角度(度): [{" ".join([f"{np.degrees(j):7.1f}" for j in self.left_target_joints])}]')
            # 左夹爪（实际：依据当前关节值反推，语义与接收一致 0=打开,1=关闭）
            try:
                _lk = self.all_joint_positions.get('left_robotiq_85_left_knuckle_joint', None)
                if _lk is not None:
                    _actual_left = 1.0 - np.clip(_lk / 0.8, 0.0, 1.0)
                else:
                    _actual_left = float(self.left_gripper_received)
            except Exception:
                _actual_left = float(self.left_gripper_received)
            self.get_logger().info(f'  左夹爪实际状态: {_actual_left:.2f}')
        
        # 右臂状态
        if self.right_target_joints is not None:
            self.get_logger().info(f'  右臂关节状态:')
            self.get_logger().info(f'  关节角度: [{" ".join([f"{j:7.3f}" for j in self.right_target_joints])}]')
            self.get_logger().info(f'  角度(度): [{" ".join([f"{np.degrees(j):7.1f}" for j in self.right_target_joints])}]')
            # 右夹爪（实际：依据当前关节值反推，语义与接收一致 0=打开,1=关闭）
            try:
                _rk = self.all_joint_positions.get('right_robotiq_85_left_knuckle_joint', None)
                if _rk is not None:
                    _actual_right = 1.0 - np.clip(_rk / 0.8, 0.0, 1.0)
                else:
                    _actual_right = float(self.right_gripper_received)
            except Exception:
                _actual_right = float(self.right_gripper_received)
            self.get_logger().info(f'  右夹爪实际状态: {_actual_right:.2f}')
        
        self.get_logger().info('='*80 + '\n')
    
    def left_ee_callback(self, msg):
        """处理左臂末端执行器目标"""
        if len(msg.data) < 3:
            self.get_logger().warn('接收到的消息数据不足3个元素')
            return
        
        rel_dx, rel_dz, rel_dyaw = [float(v) for v in msg.data[:3]]
        # 计算与上一帧相对量的增量，避免重复积分导致漂移
        last_rel = self.left_goal['last_rel']
        inc = np.array([rel_dx, rel_dz, rel_dyaw], dtype=float) - last_rel
        # 小阈值过滤（分别针对平移与角度）
        pos_inc_norm = np.hypot(inc[0], inc[1])
        if pos_inc_norm < 0.003 and abs(inc[2]) < np.deg2rad(1.0):
            return
        # 记录为新的上一帧相对量
        self.left_goal['last_rel'] = np.array([rel_dx, rel_dz, rel_dyaw], dtype=float)
        
        # 积分到绝对末端目标并夹限
        self.left_goal['x'] = float(np.clip(self.left_goal['x'] + inc[0], self.x_min, self.x_max))
        dz_raw = -inc[1] if self.invert_relative_z else inc[1]
        dz_inc = self.relative_z_gain * dz_raw
        self.left_goal['z'] = float(np.clip(self.left_goal['z'] + dz_inc, self.z_min, self.z_max))
        yaw_new = self.left_goal['yaw'] + inc[2]
        # wrap到[-pi,pi]
        yaw_new = (yaw_new + np.pi) % (2*np.pi) - np.pi
        self.left_goal['yaw'] = float(yaw_new)
        
        # 调试信息
        if not hasattr(self, '_left_callback_count'):
            self._left_callback_count = 0
        self._left_callback_count += 1
        
        # 可选解析seq（第4元素）
        seq = None
        if len(msg.data) >= 4:
            try:
                seq = int(msg.data[3])
            except Exception:
                seq = None
        # 调试：按seq或内部计数显示相对位移与偏航
        if (seq is not None and seq % self.debug_log_every_n == 0) or \
           (seq is None and self._left_callback_count % self.debug_log_every_n == 1):
            suffix = f" (seq={seq})" if seq is not None else ""
            self.get_logger().info(f'[左臂] 接收相对目标: ΔX={rel_dx:.3f}m, ΔZ={rel_dz:.3f}m, ΔYaw={rel_dyaw:.3f}rad{suffix}')
            # 在接收相对目标之后，输出左夹爪(接收)
            self.get_logger().info(f'左夹爪接收目标: {self.left_gripper_received:.2f}')
        
        # 求解逆运动学
        try:
            joint_positions = self.simple_ik_planar(
                self.left_goal['x'], self.left_goal['z'], 0.0, self.left_goal['yaw'],
                self.left_current_joints, 'left', interpret_relative=True
            )
            
            # 保存上一帧的关节状态用于对比变化
            self.left_actual_joints = self.left_current_joints.copy()
            
            # 更新当前状态
            self.left_current_joints = joint_positions
            
            # 保存目标关节状态
            self.left_target_joints = joint_positions.copy()
            
            # 更新调试计数器
            self.update_debug_counter()
            
            # 发布关节命令
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
            
            # 调试输出：发布的关节角度
            if (seq is not None and seq % self.debug_log_every_n == 0) or \
               (seq is None and self._left_callback_count % self.debug_log_every_n == 1):
                self.get_logger().info(f'[左臂] 发布关节命令: [{", ".join([f"{j:.3f}" for j in joint_positions])}]')
                joint_degrees = [f"{np.degrees(j):6.1f}°" for j in joint_positions]
                self.get_logger().info(f'         角度: [{", ".join(joint_degrees)}]')
                self.get_logger().info(f'         目标末端(x,z,yaw): ({self.left_goal["x"]:.3f}, {self.left_goal["z"]:.3f}, {self.left_goal["yaw"]:.3f})')
            
            # 发布关节状态用于RViz
            self.publish_joint_state(joint_positions, joint_names)
        except Exception as e:
            self.get_logger().error(f'左臂IK求解失败: {e}')
    
    def left_gripper_callback(self, msg):
        """处理左臂夹爪命令"""
        if len(msg.data) < 1:
            return
        
        gripper_position = msg.data[0]  # 0.0=打开, 1.0=关闭
        # 记录最近一次接收的命令值，供指定日志位置使用
        self.left_gripper_received = float(gripper_position)
        
        # 更新夹爪关节状态
        # Robotiq 85夹爪: 0.0=打开, 0.8=完全关闭
        gripper_value = (1.0 - gripper_position) * 0.8
        
        self.all_joint_positions['left_robotiq_85_left_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_right_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_left_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_right_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_left_finger_tip_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_right_finger_tip_joint'] = gripper_value
        
        # 日志按用户要求在其他位置输出
    
    def right_gripper_callback(self, msg):
        """处理右臂夹爪命令"""
        if len(msg.data) < 1:
            return
        
        gripper_position = msg.data[0]  # 0.0=打开, 1.0=关闭（反转以符合期望）
        # 记录最近一次接收的命令值
        self.right_gripper_received = float(gripper_position)
        gripper_value = (1.0 - gripper_position) * 0.8
        
        self.all_joint_positions['right_robotiq_85_left_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_right_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_left_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_right_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_left_finger_tip_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_right_finger_tip_joint'] = gripper_value
        
        # 日志按用户要求在其他位置输出
    
    def right_ee_callback(self, msg):
        """处理右臂末端执行器目标"""
        if len(msg.data) < 3:
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
        
        # 调试信息
        if not hasattr(self, '_right_callback_count'):
            self._right_callback_count = 0
        self._right_callback_count += 1
        
        # 可选解析seq（第4元素）
        seq = None
        if len(msg.data) >= 4:
            try:
                seq = int(msg.data[3])
            except Exception:
                seq = None
        # 调试：按seq或内部计数显示相对位移与偏航
        if (seq is not None and seq % self.debug_log_every_n == 0) or \
           (seq is None and self._right_callback_count % self.debug_log_every_n == 1):
            suffix = f" (seq={seq})" if seq is not None else ""
            self.get_logger().info(f'[右臂] 接收相对目标: ΔX={rel_dx:.3f}m, ΔZ={rel_dz:.3f}m, ΔYaw={rel_dyaw:.3f}rad{suffix}')
            # 在接收相对目标之后，输出右夹爪(接收)
            self.get_logger().info(f'右夹爪接收状态: {self.right_gripper_received:.2f}')
        
        # 求解逆运动学
        try:
            joint_positions = self.simple_ik_planar(
                self.right_goal['x'], self.right_goal['z'], 0.0, self.right_goal['yaw'],
                self.right_current_joints, 'right', interpret_relative=True
            )
            
            # 保存上一帧的关节状态用于对比变化
            self.right_actual_joints = self.right_current_joints.copy()
            
            # 更新当前状态
            self.right_current_joints = joint_positions
            
            # 保存目标关节状态
            self.right_target_joints = joint_positions.copy()
            
            # 发布关节命令
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
            
            # 调试输出：发布的关节角度
            if (seq is not None and seq % self.debug_log_every_n == 0) or \
               (seq is None and self._right_callback_count % self.debug_log_every_n == 1):
                self.get_logger().info(f'[右臂] 发布关节命令: [{", ".join([f"{j:.3f}" for j in joint_positions])}]')
                joint_degrees = [f"{np.degrees(j):6.1f}°" for j in joint_positions]
                self.get_logger().info(f'         角度: [{", ".join(joint_degrees)}]')
                self.get_logger().info(f'         目标末端(x,z,yaw): ({self.right_goal["x"]:.3f}, {self.right_goal["z"]:.3f}, {self.right_goal["yaw"]:.3f})')
            
            # 发布关节状态用于RViz
            self.publish_joint_state(joint_positions, joint_names)
        except Exception as e:
            self.get_logger().error(f'右臂IK求解失败: {e}')


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
