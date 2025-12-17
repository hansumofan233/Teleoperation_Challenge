#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import math
class JointAnglePublisher(Node):
    def __init__(self):
        super().__init__('joint_angle_publisher')
        
        # Subscribe to joint states
        self.joint_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Publish current joint angles - primary output
        self.all_angles_publisher = self.create_publisher(
            Float64MultiArray,
            '/current_joint_angles',  # For visualization platform
            10
        )
        
        # Publish left/right arm angles separately
        self.left_arm_publisher = self.create_publisher(
            Float64MultiArray,
            '/left_arm/joint_angles',
            10
        )
        
        self.right_arm_publisher = self.create_publisher(
            Float64MultiArray,
            '/right_arm/joint_angles',
            10
        )
        
        # Publish gripper angles
        self.gripper_publisher = self.create_publisher(
            Float64MultiArray,
            '/gripper_angles',
            10
        )
        
        # Status timer
        self.status_timer = self.create_timer(2.0, self.publish_status)
        
        # Store current angles
        self.current_joint_states = {}
        self.left_arm_joints = []
        self.right_arm_joints = []
        self.gripper_joints = []
        
        self.get_logger().info('=== Joint Angle Publisher started ===')
        self.get_logger().info('Goal: publish current joint angles from visualization as ROS messages')
        
    def joint_callback(self, msg):
        """Process joint states and publish angle data - core function"""
        try:
            # Update joint state dict
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.current_joint_states[name] = msg.position[i]
            
            # Classify joint angles
            left_angles = []
            right_angles = []
            gripper_angles = []
            all_angles = []
            
            # Process in UR3e joint order
            ur_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            
            # Left arm joints
            for joint in ur_joints:
                left_joint_name = f'left_{joint}'
                if left_joint_name in self.current_joint_states:
                    angle = self.current_joint_states[left_joint_name]
                    left_angles.append(angle)
                    all_angles.append(angle)
            
            # Right arm joints
            for joint in ur_joints:
                right_joint_name = f'right_{joint}'
                if right_joint_name in self.current_joint_states:
                    angle = self.current_joint_states[right_joint_name]
                    right_angles.append(angle)
                    all_angles.append(angle)
            
            # Gripper joints
            gripper_joint_names = ['left_robotiq_85_left_knuckle_joint', 'right_robotiq_85_left_knuckle_joint']
            for gripper_joint in gripper_joint_names:
                if gripper_joint in self.current_joint_states:
                    angle = self.current_joint_states[gripper_joint]
                    gripper_angles.append(angle)
                    all_angles.append(angle)
            
            # Publish all angles - primary target
            if all_angles:
                all_msg = Float64MultiArray()
                all_msg.data = all_angles
                self.all_angles_publisher.publish(all_msg)
            
            # Publish categorized angles
            if left_angles:
                left_msg = Float64MultiArray()
                left_msg.data = left_angles
                self.left_arm_publisher.publish(left_msg)
                self.left_arm_joints = left_angles
            
            if right_angles:
                right_msg = Float64MultiArray()
                right_msg.data = right_angles
                self.right_arm_publisher.publish(right_msg)
                self.right_arm_joints = right_angles
                
            if gripper_angles:
                gripper_msg = Float64MultiArray()
                gripper_msg.data = gripper_angles
                self.gripper_publisher.publish(gripper_msg)
                self.gripper_joints = gripper_angles
                
        except Exception as e:
            self.get_logger().error(f'Error processing joint states: {e}')
    
    def publish_status(self):
        """Publish status periodically"""
        if self.left_arm_joints and self.right_arm_joints:
            # Convert to degrees for display
            left_degrees = [math.degrees(angle) for angle in self.left_arm_joints]
            right_degrees = [math.degrees(angle) for angle in self.right_arm_joints]
            
            self.get_logger().info(f'Left arm joint angles (deg): {[f"{a:.1f}" for a in left_degrees]}')
            self.get_logger().info(f'Right arm joint angles (deg): {[f"{a:.1f}" for a in right_degrees]}')
            
            if self.gripper_joints:
                gripper_degrees = [math.degrees(angle) for angle in self.gripper_joints]
                self.get_logger().info(f'Gripper angles (deg): {[f"{a:.1f}" for a in gripper_degrees]}')
        else:
            self.get_logger().warn('Waiting for joint state data...')
def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = JointAnglePublisher()
        node.get_logger().info('Publishing joint angle data...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down Joint Angle Publisher...')
    except Exception as e:
        print(f'Node runtime error: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()