#!/usr/bin/env python3
"""
末端执行器控制测试工具
用于测试和调试摄像头遥操作系统
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory
import time

class TeleopTester(Node):
    def __init__(self):
        super().__init__('teleop_tester')
        
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
        
        # 订阅关节轨迹
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
        
        # 统计信息
        self.left_ee_count = 0
        self.right_ee_count = 0
        self.left_joint_count = 0
        self.right_joint_count = 0
        
        self.start_time = time.time()
        
        # 创建定时器打印状态
        self.timer = self.create_timer(1.0, self.print_status)
        
        self.get_logger().info('遥操作测试工具已启动')
        self.get_logger().info('开始监听话题...')
    
    def left_ee_callback(self, msg):
        """左臂末端执行器回调"""
        self.left_ee_count += 1
        if len(msg.data) >= 4:
            x, y, z, rot = msg.data[:4]
            self.get_logger().info(
                f'左臂EE目标: X={x:.3f}, Y={y:.3f}, Z={z:.3f}, Rot={rot:.3f}rad',
                throttle_duration_sec=2.0
            )
    
    def right_ee_callback(self, msg):
        """右臂末端执行器回调"""
        self.right_ee_count += 1
        if len(msg.data) >= 4:
            x, y, z, rot = msg.data[:4]
            self.get_logger().info(
                f'右臂EE目标: X={x:.3f}, Y={y:.3f}, Z={z:.3f}, Rot={rot:.3f}rad',
                throttle_duration_sec=2.0
            )
    
    def left_joint_callback(self, msg):
        """左臂关节轨迹回调"""
        self.left_joint_count += 1
        if msg.points:
            positions = msg.points[0].positions
            self.get_logger().debug(
                f'左臂关节: {[f"{p:.3f}" for p in positions]}'
            )
    
    def right_joint_callback(self, msg):
        """右臂关节轨迹回调"""
        self.right_joint_count += 1
        if msg.points:
            positions = msg.points[0].positions
            self.get_logger().debug(
                f'右臂关节: {[f"{p:.3f}" for p in positions]}'
            )
    
    def print_status(self):
        """打印统计状态"""
        elapsed = time.time() - self.start_time
        
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'运行时间: {elapsed:.1f}秒')
        self.get_logger().info(f'左臂EE消息: {self.left_ee_count} (速率: {self.left_ee_count/elapsed:.1f} Hz)')
        self.get_logger().info(f'右臂EE消息: {self.right_ee_count} (速率: {self.right_ee_count/elapsed:.1f} Hz)')
        self.get_logger().info(f'左臂关节消息: {self.left_joint_count} (速率: {self.left_joint_count/elapsed:.1f} Hz)')
        self.get_logger().info(f'右臂关节消息: {self.right_joint_count} (速率: {self.right_joint_count/elapsed:.1f} Hz)')
        self.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    
    node = TeleopTester()
    
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
