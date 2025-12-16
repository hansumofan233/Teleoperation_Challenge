#!/usr/bin/env python3
"""
数据流验证工具
实时监控teleop和commander节点之间的数据传输
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory
import numpy as np
from collections import deque
import time

class DataFlowVerifier(Node):
    def __init__(self):
        super().__init__('data_flow_verifier')
        
        # 订阅末端执行器目标（teleop发送的）
        self.left_target_sub = self.create_subscription(
            Float64MultiArray,
            '/left_arm/endeffector_target',
            self.left_target_callback,
            10
        )
        self.right_target_sub = self.create_subscription(
            Float64MultiArray,
            '/right_arm/endeffector_target',
            self.right_target_callback,
            10
        )
        
        # 订阅关节轨迹命令（commander发送的）
        self.left_joint_sub = self.create_subscription(
            JointTrajectory,
            '/left_arm/joint_trajectory_controller/joint_trajectory',
            self.left_joint_callback,
            10
        )
        self.right_joint_sub = self.create_subscription(
            JointTrajectory,
            '/right_arm/joint_trajectory_controller/joint_trajectory',
            self.right_joint_callback,
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
        
        # 数据存储
        self.left_targets = deque(maxlen=100)
        self.right_targets = deque(maxlen=100)
        self.left_joints = deque(maxlen=100)
        self.right_joints = deque(maxlen=100)
        
        # 统计信息
        self.stats = {
            'left_target_count': 0,
            'right_target_count': 0,
            'left_joint_count': 0,
            'right_joint_count': 0,
            'left_gripper_count': 0,
            'right_gripper_count': 0,
            'last_left_target': None,
            'last_right_target': None,
            'last_left_joint': None,
            'last_right_joint': None,
        }
        
        # 创建定时器定期显示统计
        self.timer = self.create_timer(2.0, self.print_statistics)
        
        self.get_logger().info('=' * 80)
        self.get_logger().info('数据流验证工具已启动')
        self.get_logger().info('监控话题:')
        self.get_logger().info('  📤 /left_arm/endeffector_target (teleop → commander)')
        self.get_logger().info('  📤 /right_arm/endeffector_target (teleop → commander)')
        self.get_logger().info('  📥 /left_arm/joint_trajectory_controller/joint_trajectory (commander → robot)')
        self.get_logger().info('  📥 /right_arm/joint_trajectory_controller/joint_trajectory (commander → robot)')
        self.get_logger().info('=' * 80)
    
    def left_target_callback(self, msg):
        """接收左臂末端执行器目标"""
        self.stats['left_target_count'] += 1
        self.stats['last_left_target'] = msg.data
        self.left_targets.append((time.time(), msg.data))
    
    def right_target_callback(self, msg):
        """接收右臂末端执行器目标"""
        self.stats['right_target_count'] += 1
        self.stats['last_right_target'] = msg.data
        self.right_targets.append((time.time(), msg.data))
    
    def left_joint_callback(self, msg):
        """接收左臂关节命令"""
        self.stats['left_joint_count'] += 1
        if msg.points:
            self.stats['last_left_joint'] = msg.points[0].positions
            self.left_joints.append((time.time(), msg.points[0].positions))
    
    def right_joint_callback(self, msg):
        """接收右臂关节命令"""
        self.stats['right_joint_count'] += 1
        if msg.points:
            self.stats['last_right_joint'] = msg.points[0].positions
            self.right_joints.append((time.time(), msg.points[0].positions))
    
    def left_gripper_callback(self, msg):
        """接收左臂夹爪命令"""
        self.stats['left_gripper_count'] += 1
        gripper_state = "打开" if msg.data[0] > 0.5 else "关闭"
        self.get_logger().info(f'🤏 [左夹爪] {gripper_state} (值: {msg.data[0]:.2f})')
    
    def right_gripper_callback(self, msg):
        """接收右臂夹爪命令"""
        self.stats['right_gripper_count'] += 1
        gripper_state = "打开" if msg.data[0] > 0.5 else "关闭"
        self.get_logger().info(f'🤏 [右夹爪] {gripper_state} (值: {msg.data[0]:.2f})')
    
    def print_statistics(self):
        """打印统计信息"""
        current_time = time.time()
        
        self.get_logger().info('\n' + '=' * 80)
        self.get_logger().info('📊 数据流统计（实时）')
        self.get_logger().info('=' * 80)
        
        # 左臂数据流
        self.get_logger().info('\n【左臂数据流】')
        self.get_logger().info(f'  末端目标消息数: {self.stats["left_target_count"]}')
        self.get_logger().info(f'  关节命令消息数: {self.stats["left_joint_count"]}')
        self.get_logger().info(f'  夹爪命令消息数: {self.stats["left_gripper_count"]}')
        
        if self.stats['last_left_target']:
            target = self.stats['last_left_target']
            self.get_logger().info(f'  📍 最新目标: X={target[0]:.3f}, Y={target[1]:.3f}, Z={target[2]:.3f}')
        
        if self.stats['last_left_joint']:
            joints = self.stats['last_left_joint']
            joint_degrees = [f'{np.degrees(j):6.1f}°' for j in joints]
            self.get_logger().info(f'  🔧 最新关节: [{", ".join(joint_degrees)}]')
        
        # 计算左臂更新频率
        if len(self.left_targets) > 1:
            recent_targets = [t for t, _ in self.left_targets if current_time - t < 2.0]
            freq = len(recent_targets) / 2.0
            self.get_logger().info(f'  📈 目标更新频率: {freq:.1f} Hz')
        
        # 右臂数据流
        self.get_logger().info('\n【右臂数据流】')
        self.get_logger().info(f'  末端目标消息数: {self.stats["right_target_count"]}')
        self.get_logger().info(f'  关节命令消息数: {self.stats["right_joint_count"]}')
        self.get_logger().info(f'  夹爪命令消息数: {self.stats["right_gripper_count"]}')
        
        if self.stats['last_right_target']:
            target = self.stats['last_right_target']
            self.get_logger().info(f'  📍 最新目标: X={target[0]:.3f}, Y={target[1]:.3f}, Z={target[2]:.3f}')
        
        if self.stats['last_right_joint']:
            joints = self.stats['last_right_joint']
            joint_degrees = [f'{np.degrees(j):6.1f}°' for j in joints]
            self.get_logger().info(f'  🔧 最新关节: [{", ".join(joint_degrees)}]')
        
        # 计算右臂更新频率
        if len(self.right_targets) > 1:
            recent_targets = [t for t, _ in self.right_targets if current_time - t < 2.0]
            freq = len(recent_targets) / 2.0
            self.get_logger().info(f'  📈 目标更新频率: {freq:.1f} Hz')
        
        # 数据一致性检查
        self.get_logger().info('\n【数据一致性】')
        if self.stats['left_target_count'] > 0 and self.stats['left_joint_count'] > 0:
            ratio = self.stats['left_joint_count'] / self.stats['left_target_count']
            status = '✅ 正常' if 0.8 <= ratio <= 1.2 else '⚠️  异常'
            self.get_logger().info(f'  左臂: 目标→关节 比例 {ratio:.2f} {status}')
        
        if self.stats['right_target_count'] > 0 and self.stats['right_joint_count'] > 0:
            ratio = self.stats['right_joint_count'] / self.stats['right_target_count']
            status = '✅ 正常' if 0.8 <= ratio <= 1.2 else '⚠️  异常'
            self.get_logger().info(f'  右臂: 目标→关节 比例 {ratio:.2f} {status}')
        
        self.get_logger().info('=' * 80 + '\n')


def main(args=None):
    rclpy.init(args=args)
    
    node = DataFlowVerifier()
    
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
