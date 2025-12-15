#!/usr/bin/env python3
"""
关节角度API - 为远程操作平台提供标准化的关节角度接口
适用于集成到Web平台或其他可视化系统
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import json
import math

class JointAngleAPI(Node):
    def __init__(self):
        super().__init__('joint_angle_api')
        
        # 订阅关节状态
        self.joint_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # 发布标准化的关节角度数据
        self.angles_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot/joint_angles',
            10
        )
        
        # 发布JSON格式的数据（便于Web集成）
        self.json_publisher = self.create_publisher(
            String,
            '/robot/joint_angles_json',
            10
        )
        
        # 发布结构化数据
        self.structured_publisher = self.create_publisher(
            String,
            '/robot/structured_data',
            10
        )
        
        # 定时发布状态
        self.status_timer = self.create_timer(0.1, self.publish_current_state)  # 10Hz
        
        # 数据存储
        self.current_angles = {
            'left_arm': [0.0] * 6,
            'right_arm': [0.0] * 6,
            'grippers': [0.0] * 2,
            'timestamp': 0.0,
            'valid': False
        }
        
        self.get_logger().info('=== 关节角度API已启动 ===')
        self.get_logger().info('发布话题:')
        self.get_logger().info('  /robot/joint_angles - 数组格式')
        self.get_logger().info('  /robot/joint_angles_json - JSON格式')
        self.get_logger().info('  /robot/structured_data - 结构化数据')
        
    def joint_callback(self, msg):
        """处理关节状态数据"""
        try:
            # 构建关节字典
            joint_dict = {}
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    joint_dict[name] = msg.position[i]
            
            # 提取左臂角度
            left_arm = []
            ur_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            
            for joint in ur_joints:
                left_joint = f'left_{joint}'
                if left_joint in joint_dict:
                    left_arm.append(joint_dict[left_joint])
            
            # 提取右臂角度
            right_arm = []
            for joint in ur_joints:
                right_joint = f'right_{joint}'
                if right_joint in joint_dict:
                    right_arm.append(joint_dict[right_joint])
            
            # 提取夹爪角度
            grippers = []
            gripper_joints = ['left_robotiq_85_left_knuckle_joint', 'right_robotiq_85_left_knuckle_joint']
            for gripper in gripper_joints:
                if gripper in joint_dict:
                    grippers.append(joint_dict[gripper])
            
            # 更新数据
            if len(left_arm) == 6 and len(right_arm) == 6:
                self.current_angles.update({
                    'left_arm': left_arm,
                    'right_arm': right_arm,
                    'grippers': grippers if len(grippers) == 2 else [0.0, 0.0],
                    'timestamp': self.get_clock().now().nanoseconds / 1e9,
                    'valid': True
                })
                
        except Exception as e:
            self.get_logger().error(f'处理关节数据出错: {e}')
            self.current_angles['valid'] = False
    
    def publish_current_state(self):
        """发布当前状态到所有话题"""
        if not self.current_angles['valid']:
            return
            
        try:
            # 1. 发布数组格式 - 标准ROS消息
            all_angles = (self.current_angles['left_arm'] + 
                         self.current_angles['right_arm'] + 
                         self.current_angles['grippers'])
            
            angle_msg = Float64MultiArray()
            angle_msg.data = all_angles
            self.angles_publisher.publish(angle_msg)
            
            # 2. 发布JSON格式 - 便于Web集成
            json_data = {
                'left_arm': {
                    'angles_rad': self.current_angles['left_arm'],
                    'angles_deg': [math.degrees(a) for a in self.current_angles['left_arm']],
                    'joint_names': ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
                },
                'right_arm': {
                    'angles_rad': self.current_angles['right_arm'],
                    'angles_deg': [math.degrees(a) for a in self.current_angles['right_arm']],
                    'joint_names': ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
                },
                'grippers': {
                    'angles_rad': self.current_angles['grippers'],
                    'angles_deg': [math.degrees(a) for a in self.current_angles['grippers']],
                    'names': ['left_gripper', 'right_gripper']
                },
                'timestamp': self.current_angles['timestamp'],
                'total_joints': len(all_angles)
            }
            
            json_msg = String()
            json_msg.data = json.dumps(json_data, indent=2)
            self.json_publisher.publish(json_msg)
            
            # 3. 发布结构化数据 - 便于调试和监控
            structured_data = {
                'status': 'active',
                'dual_arm_angles': {
                    'left': [f"{a:.3f}" for a in self.current_angles['left_arm']],
                    'right': [f"{a:.3f}" for a in self.current_angles['right_arm']],
                    'grippers': [f"{a:.3f}" for a in self.current_angles['grippers']]
                },
                'summary': {
                    'left_arm_deg': [f"{math.degrees(a):.1f}°" for a in self.current_angles['left_arm']],
                    'right_arm_deg': [f"{math.degrees(a):.1f}°" for a in self.current_angles['right_arm']]
                }
            }
            
            struct_msg = String()
            struct_msg.data = json.dumps(structured_data, indent=2)
            self.structured_publisher.publish(struct_msg)
            
        except Exception as e:
            self.get_logger().error(f'发布数据出错: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = JointAngleAPI()
        node.get_logger().info('关节角度API就绪，开始发布数据...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n关节角度API已停止')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
