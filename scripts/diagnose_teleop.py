#!/usr/bin/env python3
"""
快速诊断工具 - 检查摄像头控制节点状态
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

class DiagnosticTool(Node):
    def __init__(self):
        super().__init__('diagnostic_tool')
        
        self.left_count = 0
        self.right_count = 0
        self.last_left_data = None
        self.last_right_data = None
        
        self.left_sub = self.create_subscription(
            Float64MultiArray,
            '/left_arm/endeffector_target',
            self.left_callback,
            10
        )
        
        self.right_sub = self.create_subscription(
            Float64MultiArray,
            '/right_arm/endeffector_target',
            self.right_callback,
            10
        )
        
        self.timer = self.create_timer(2.0, self.print_status)
        self.start_time = time.time()
        
        print("\n" + "="*60)
        print("诊断工具已启动 - 监听末端执行器目标话题")
        print("="*60)
        print("如果5秒内没有消息,可能的原因:")
        print("1. 摄像头检测不到手臂(确保整个手臂在画面内)")
        print("2. 手臂可见性太低(改善光线)")
        print("3. 需要先按'c'键校准")
        print("="*60 + "\n")
    
    def left_callback(self, msg):
        self.left_count += 1
        self.last_left_data = msg.data
    
    def right_callback(self, msg):
        self.right_count += 1
        self.last_right_data = msg.data
    
    def print_status(self):
        elapsed = time.time() - self.start_time
        
        print(f"\n[{elapsed:.1f}s] 状态:")
        print(f"  左臂消息数: {self.left_count}")
        if self.last_left_data and len(self.last_left_data) >= 4:
            print(f"    → 最新位置: X={self.last_left_data[0]:.3f}, Y={self.last_left_data[1]:.3f}, Z={self.last_left_data[2]:.3f}")
            print(f"    → 旋转: {self.last_left_data[3]:.3f} rad")
        
        print(f"  右臂消息数: {self.right_count}")
        if self.last_right_data and len(self.last_right_data) >= 4:
            print(f"    → 最新位置: X={self.last_right_data[0]:.3f}, Y={self.last_right_data[1]:.3f}, Z={self.last_right_data[2]:.3f}")
            print(f"    → 旋转: {self.last_right_data[3]:.3f} rad")
        
        if self.left_count == 0 and self.right_count == 0:
            print("  ⚠️  没有收到任何数据!")
            print("  → 检查摄像头窗口,确认手臂是否被检测到")
            print("  → 尝试按'c'键进行校准")


def main():
    rclpy.init()
    node = DiagnosticTool()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n诊断结束")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
