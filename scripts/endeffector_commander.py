#!/usr/bin/env python3
"""
末端执行器命令转换节点
将末端执行器目标位置转换为关节命令
使用简化的逆运动学求解
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.optimize import minimize

class EndEffectorCommander(Node):
    def __init__(self):
        super().__init__('endeffector_commander')
        
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
            'left_shoulder_lift_joint': -np.pi/4,
            'left_elbow_joint': np.pi/2,
            'left_wrist_1_joint': -np.pi/4,
            'left_wrist_2_joint': -np.pi/2,
            'left_wrist_3_joint': 0.0,
            'right_shoulder_pan_joint': 0.0,
            'right_shoulder_lift_joint': -np.pi/4,
            'right_elbow_joint': np.pi/2,
            'right_wrist_1_joint': -np.pi/4,
            'right_wrist_2_joint': -np.pi/2,
            'right_wrist_3_joint': 0.0,
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
        self.joint_limits = [
            (-2*np.pi, 2*np.pi),  # 基座
            (-np.pi, np.pi),       # 肩部
            (-np.pi, np.pi),       # 肘部
            (-2*np.pi, 2*np.pi),  # 腕1
            (-2*np.pi, 2*np.pi),  # 腕2
            (-2*np.pi, 2*np.pi),  # 腕3
        ]
        
        # 当前关节状态(用于连续性)
        self.left_current_joints = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0.0])
        self.right_current_joints = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0.0])
        
        self.get_logger().info('末端执行器命令转换节点已启动')
    
    def simple_ik_planar(self, target_x, target_y, target_z, target_rot_z, current_joints):
        """
        简化的逆运动学求解(针对平面运动+旋转)
        
        策略:
        - 固定肩部和肘部关节实现平面XY位置
        - 使用腕部关节实现旋转
        """
        
        def forward_kinematics_simplified(joints):
            """简化正运动学(只计算末端位置和方向)"""
            q1, q2, q3, q4, q5, q6 = joints
            
            # 简化计算: 只关注XY平面和Z轴旋转
            # 基座旋转
            x = (self.a2 * np.cos(q2) + self.a3 * np.cos(q2 + q3)) * np.cos(q1)
            y = (self.a2 * np.cos(q2) + self.a3 * np.cos(q2 + q3)) * np.sin(q1)
            z = self.d1 + self.a2 * np.sin(q2) + self.a3 * np.sin(q2 + q3)
            
            # 末端方向(简化为绕Z轴)
            rot_z = q1 + q4 + q5 + q6
            
            return np.array([x, y, z, rot_z])
        
        def cost_function(joints):
            """优化目标函数"""
            fk_result = forward_kinematics_simplified(joints)
            
            # 位置误差
            pos_error = np.sum((fk_result[:3] - np.array([target_x, target_y, target_z]))**2)
            
            # 旋转误差
            rot_error = (fk_result[3] - target_rot_z)**2
            
            # 关节变化惩罚(保持平滑)
            joint_change = np.sum((joints - current_joints)**2) * 0.1
            
            return pos_error + rot_error * 0.5 + joint_change
        
        # 使用优化求解
        result = minimize(
            cost_function,
            current_joints,
            method='SLSQP',
            bounds=self.joint_limits,
            options={'maxiter': 100, 'ftol': 1e-4}
        )
        
        if result.success:
            return result.x
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
        """发布所有关节状态用于RViz可视化"""
        # 更新对应关节的位置
        for i, name in enumerate(joint_names):
            if name in self.all_joint_positions:
                self.all_joint_positions[name] = joint_positions[i]
        
        # 发布完整的关节状态
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.all_joint_positions.keys())
        msg.position = list(self.all_joint_positions.values())
        msg.velocity = []
        msg.effort = []
        
        self.joint_state_pub.publish(msg)
    
    def left_ee_callback(self, msg):
        """处理左臂末端执行器目标"""
        if len(msg.data) < 3:
            self.get_logger().warn('接收到的消息数据不足3个元素')
            return
        
        target_x, target_y, target_z = msg.data[:3]
        target_rot_z = 0.0  # 默认旋转角度
        
        # 调试信息
        if not hasattr(self, '_left_callback_count'):
            self._left_callback_count = 0
        self._left_callback_count += 1
        
        if self._left_callback_count % 30 == 1:
            self.get_logger().info(f'左臂目标: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}, Rot={target_rot_z:.3f}')
        
        # 求解逆运动学
        try:
            joint_positions = self.simple_ik_planar(
                target_x, target_y, target_z, target_rot_z,
                self.left_current_joints
            )
            
            # 更新当前状态
            self.left_current_joints = joint_positions
            
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
            
            # 发布关节状态用于RViz
            self.publish_joint_state(joint_positions, joint_names)
            
            if self._left_callback_count % 30 == 1:
                self.get_logger().info(f'已发布左臂关节命令')
        except Exception as e:
            self.get_logger().error(f'左臂IK求解失败: {e}')
    
    def left_gripper_callback(self, msg):
        """处理左臂夹爪命令"""
        if len(msg.data) < 1:
            return
        
        gripper_position = msg.data[0]  # 0.0=关闭, 1.0=打开
        
        # 更新夹爪关节状态
        # Robotiq 85夹爪: 0.0=关闭, 0.8=完全打开
        gripper_value = gripper_position * 0.8
        
        self.all_joint_positions['left_robotiq_85_left_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_right_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_left_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_right_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_left_finger_tip_joint'] = gripper_value
        self.all_joint_positions['left_robotiq_85_right_finger_tip_joint'] = gripper_value
        
        self.get_logger().info(f'左夹爪: {gripper_position:.2f}')
    
    def right_gripper_callback(self, msg):
        """处理右臂夹爪命令"""
        if len(msg.data) < 1:
            return
        
        gripper_position = msg.data[0]
        gripper_value = gripper_position * 0.8
        
        self.all_joint_positions['right_robotiq_85_left_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_right_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_left_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_right_inner_knuckle_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_left_finger_tip_joint'] = gripper_value
        self.all_joint_positions['right_robotiq_85_right_finger_tip_joint'] = gripper_value
        
        self.get_logger().info(f'右夹爪: {gripper_position:.2f}')
    
    def right_ee_callback(self, msg):
        """处理右臂末端执行器目标"""
        if len(msg.data) < 3:
            return
        
        target_x, target_y, target_z = msg.data[:3]
        target_rot_z = 0.0
        
        # 调试信息
        if not hasattr(self, '_right_callback_count'):
            self._right_callback_count = 0
        self._right_callback_count += 1
        
        if self._right_callback_count % 30 == 1:
            self.get_logger().info(f'右臂目标: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}, Rot={target_rot_z:.3f}')
        
        # 求解逆运动学
        try:
            joint_positions = self.simple_ik_planar(
                target_x, target_y, target_z, target_rot_z,
                self.right_current_joints
            )
            
            # 更新当前状态
            self.right_current_joints = joint_positions
            
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
            
            # 发布关节状态用于RViz
            self.publish_joint_state(joint_positions, joint_names)
            
            if self._right_callback_count % 30 == 1:
                self.get_logger().info(f'已发布右臂关节命令')
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
