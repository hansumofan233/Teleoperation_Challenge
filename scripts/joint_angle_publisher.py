#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import math
class JointAnglePublisher(Node):
    def __init__(self):
        super().__init__('joint_angle_publisher')
        
        # 订阅关节状态
        self.joint_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # 发布当前关节角度 - 这是您的最终目标
        self.all_angles_publisher = self.create_publisher(
            Float64MultiArray,
            '/current_joint_angles',  # 用于可视化平台
            10
        )
        
        # 分别发布左右臂角度
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
        
        # 发布夹爪角度
        self.gripper_publisher = self.create_publisher(
            Float64MultiArray,
            '/gripper_angles',
            10
        )
        
        # 状态定时器
        self.status_timer = self.create_timer(2.0, self.publish_status)
        
        # 存储当前角度
        self.current_joint_states = {}
        self.left_arm_joints = []
        self.right_arm_joints = []
        self.gripper_joints = []
        
        self.get_logger().info('=== 关节角度发布器已启动 ===')
        self.get_logger().info('目标: 将可视化平台上的当前关节角度值发布为ROS消息')
        
    def joint_callback(self, msg):
        """处理关节状态并发布角度数据 - 核心功能"""
        try:
            # 更新关节状态字典
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.current_joint_states[name] = msg.position[i]
            
            # 分类关节角度
            left_angles = []
            right_angles = []
            gripper_angles = []
            all_angles = []
            
            # 按照UR3e关节顺序处理
            ur_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            
            # 处理左臂关节
            for joint in ur_joints:
                left_joint_name = f'left_{joint}'
                if left_joint_name in self.current_joint_states:
                    angle = self.current_joint_states[left_joint_name]
                    left_angles.append(angle)
                    all_angles.append(angle)
            
            # 处理右臂关节
            for joint in ur_joints:
                right_joint_name = f'right_{joint}'
                if right_joint_name in self.current_joint_states:
                    angle = self.current_joint_states[right_joint_name]
                    right_angles.append(angle)
                    all_angles.append(angle)
            
            # 处理夹爪关节
            gripper_joint_names = ['left_robotiq_85_left_knuckle_joint', 'right_robotiq_85_left_knuckle_joint']
            for gripper_joint in gripper_joint_names:
                if gripper_joint in self.current_joint_states:
                    angle = self.current_joint_states[gripper_joint]
                    gripper_angles.append(angle)
                    all_angles.append(angle)
            
            # 发布所有角度 - 这是您的主要目标
            if all_angles:
                all_msg = Float64MultiArray()
                all_msg.data = all_angles
                self.all_angles_publisher.publish(all_msg)
            
            # 发布分类角度
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
            self.get_logger().error(f'处理关节状态时出错: {e}')
    
    def publish_status(self):
        """定期发布状态信息"""
        if self.left_arm_joints and self.right_arm_joints:
            # 转换为度数显示
            left_degrees = [math.degrees(angle) for angle in self.left_arm_joints]
            right_degrees = [math.degrees(angle) for angle in self.right_arm_joints]
            
            self.get_logger().info(f'左臂关节角度(度): {[f"{a:.1f}" for a in left_degrees]}')
            self.get_logger().info(f'右臂关节角度(度): {[f"{a:.1f}" for a in right_degrees]}')
            
            if self.gripper_joints:
                gripper_degrees = [math.degrees(angle) for angle in self.gripper_joints]
                self.get_logger().info(f'夹爪角度(度): {[f"{a:.1f}" for a in gripper_degrees]}')
        else:
            self.get_logger().warn('等待关节状态数据...')
def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = JointAnglePublisher()
        node.get_logger().info('开始发布关节角度数据...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n正在关闭关节角度发布器...')
    except Exception as e:
        print(f'节点运行出错: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()