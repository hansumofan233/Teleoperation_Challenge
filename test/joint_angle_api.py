#!/usr/bin/env python3
"""
Joint Angle API - standardized joint angle interface for teleoperation platforms
Suitable for integration with web platforms or other visualization systems
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
        
        # Subscribe to joint states
        self.joint_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Publish standardized joint angle data
        self.angles_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot/joint_angles',
            10
        )
        
        # Publish JSON data (friendly for web integration)
        self.json_publisher = self.create_publisher(
            String,
            '/robot/joint_angles_json',
            10
        )
        
        # Publish structured data
        self.structured_publisher = self.create_publisher(
            String,
            '/robot/structured_data',
            10
        )
        
        # Periodically publish status
        self.status_timer = self.create_timer(0.1, self.publish_current_state)  # 10Hz
        
        # Data storage
        self.current_angles = {
            'left_arm': [0.0] * 6,
            'right_arm': [0.0] * 6,
            'grippers': [0.0] * 2,
            'timestamp': 0.0,
            'valid': False
        }
        
        self.get_logger().info('=== Joint Angle API started ===')
        self.get_logger().info('Topics:')
        self.get_logger().info('  /robot/joint_angles - array format')
        self.get_logger().info('  /robot/joint_angles_json - JSON format')
        self.get_logger().info('  /robot/structured_data - structured data')
        
    def joint_callback(self, msg):
        """Process joint state data"""
        try:
            # Build joint dictionary
            joint_dict = {}
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    joint_dict[name] = msg.position[i]
            
            # Extract left arm angles
            left_arm = []
            ur_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            
            for joint in ur_joints:
                left_joint = f'left_{joint}'
                if left_joint in joint_dict:
                    left_arm.append(joint_dict[left_joint])
            
            # Extract right arm angles
            right_arm = []
            for joint in ur_joints:
                right_joint = f'right_{joint}'
                if right_joint in joint_dict:
                    right_arm.append(joint_dict[right_joint])
            
            # Extract gripper angles
            grippers = []
            gripper_joints = ['left_robotiq_85_left_knuckle_joint', 'right_robotiq_85_left_knuckle_joint']
            for gripper in gripper_joints:
                if gripper in joint_dict:
                    grippers.append(joint_dict[gripper])
            
            # Update data
            if len(left_arm) == 6 and len(right_arm) == 6:
                self.current_angles.update({
                    'left_arm': left_arm,
                    'right_arm': right_arm,
                    'grippers': grippers if len(grippers) == 2 else [0.0, 0.0],
                    'timestamp': self.get_clock().now().nanoseconds / 1e9,
                    'valid': True
                })
                
        except Exception as e:
            self.get_logger().error(f'Error processing joint data: {e}')
            self.current_angles['valid'] = False
    
    def publish_current_state(self):
        """Publish current state to all topics"""
        if not self.current_angles['valid']:
            return
            
        try:
            # 1. Publish array format - standard ROS message
            all_angles = (self.current_angles['left_arm'] + 
                         self.current_angles['right_arm'] + 
                         self.current_angles['grippers'])
            
            angle_msg = Float64MultiArray()
            angle_msg.data = all_angles
            self.angles_publisher.publish(angle_msg)
            
            # 2. Publish JSON format - for web integration
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
            
            # 3. Publish structured data - for debugging and monitoring
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
            self.get_logger().error(f'Error publishing data: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = JointAngleAPI()
        node.get_logger().info('Joint Angle API ready, starting to publish data...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nJoint Angle API stopped')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
