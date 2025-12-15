#!/usr/bin/env python3
"""
摄像头遥操作末端执行器控制节点
实现2D平面运动 + 1D旋转控制
使用MediaPipe捕获手臂运动并映射到机械臂末端执行器
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float64MultiArray, Header
import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class TeleopEndEffectorControl(Node):
    def __init__(self):
        super().__init__('teleop_endeffector_control')
        
        # ROS2 Publishers - 发布末端执行器目标位置
        self.left_ee_pub = self.create_publisher(
            Float64MultiArray, 
            '/left_arm/endeffector_target', 
            10
        )
        self.right_ee_pub = self.create_publisher(
            Float64MultiArray, 
            '/right_arm/endeffector_target', 
            10
        )
        
        # 参数配置
        self.declare_parameter('workspace_x_min', 0.2)  # 工作空间X范围 (m)
        self.declare_parameter('workspace_x_max', 0.8)
        self.declare_parameter('workspace_y_min', -0.3)  # 工作空间Y范围 (m)
        self.declare_parameter('workspace_y_max', 0.3)
        self.declare_parameter('workspace_z', 0.9)  # 固定Z高度 (m)
        self.declare_parameter('rotation_sensitivity', 2.0)  # 旋转灵敏度
        self.declare_parameter('position_smoothing', 0.3)  # 位置平滑系数
        self.declare_parameter('camera_id', 0)
        
        # 获取参数
        self.workspace_x_min = self.get_parameter('workspace_x_min').value
        self.workspace_x_max = self.get_parameter('workspace_x_max').value
        self.workspace_y_min = self.get_parameter('workspace_y_min').value
        self.workspace_y_max = self.get_parameter('workspace_y_max').value
        self.workspace_z = self.get_parameter('workspace_z').value
        self.rotation_sensitivity = self.get_parameter('rotation_sensitivity').value
        self.smoothing = self.get_parameter('position_smoothing').value
        camera_id = self.get_parameter('camera_id').value
        
        # 初始化MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 关键点索引
        self.keypoints = {
            'LEFT_SHOULDER': 11,
            'LEFT_ELBOW': 13,
            'LEFT_WRIST': 15,
            'RIGHT_SHOULDER': 12,
            'RIGHT_ELBOW': 14,
            'RIGHT_WRIST': 16,
        }
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 {camera_id}')
            raise RuntimeError('摄像头初始化失败')
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 状态变量
        self.left_arm_state = {
            'position': np.array([0.5, 0.3, self.workspace_z]),  # 左臂初始位置
            'rotation': 0.0,
            'prev_position': None,
            'prev_rotation': None
        }
        self.right_arm_state = {
            'position': np.array([0.5, -0.3, self.workspace_z]),  # 右臂初始位置
            'rotation': 0.0,
            'prev_position': None,
            'prev_rotation': None
        }
        
        # 校准参考点(用于建立映射关系)
        self.calibration_mode = False
        self.calibration_data = {
            'left': {'shoulder': None, 'reference_wrist': None},
            'right': {'shoulder': None, 'reference_wrist': None}
        }
        
        # 创建定时器处理视频流
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30Hz
        
        self.get_logger().info('末端执行器遥操作控制节点已启动')
        self.get_logger().info(f'工作空间: X[{self.workspace_x_min}, {self.workspace_x_max}], '
                              f'Y[{self.workspace_y_min}, {self.workspace_y_max}], '
                              f'Z={self.workspace_z}')
        self.get_logger().info('按键说明: q-退出, c-校准, r-重置, s-保存截图')
    
    def calibrate(self, pose_landmarks):
        """校准模式: 记录T姿态作为参考"""
        landmarks = pose_landmarks.landmark
        
        # 记录左臂参考点
        if landmarks[self.keypoints['LEFT_WRIST']].visibility > 0.5:
            self.calibration_data['left']['shoulder'] = np.array([
                landmarks[self.keypoints['LEFT_SHOULDER']].x,
                landmarks[self.keypoints['LEFT_SHOULDER']].y,
                landmarks[self.keypoints['LEFT_SHOULDER']].z
            ])
            self.calibration_data['left']['reference_wrist'] = np.array([
                landmarks[self.keypoints['LEFT_WRIST']].x,
                landmarks[self.keypoints['LEFT_WRIST']].y,
                landmarks[self.keypoints['LEFT_WRIST']].z
            ])
        
        # 记录右臂参考点
        if landmarks[self.keypoints['RIGHT_WRIST']].visibility > 0.5:
            self.calibration_data['right']['shoulder'] = np.array([
                landmarks[self.keypoints['RIGHT_SHOULDER']].x,
                landmarks[self.keypoints['RIGHT_SHOULDER']].y,
                landmarks[self.keypoints['RIGHT_SHOULDER']].z
            ])
            self.calibration_data['right']['reference_wrist'] = np.array([
                landmarks[self.keypoints['RIGHT_WRIST']].x,
                landmarks[self.keypoints['RIGHT_WRIST']].y,
                landmarks[self.keypoints['RIGHT_WRIST']].z
            ])
        
        self.get_logger().info('校准完成! 当前姿态已设为参考位置')
    
    def calculate_endeffector_pose(self, pose_landmarks, arm='left'):
        """
        计算末端执行器位姿 (2D位置 + 1D旋转)
        
        映射策略:
        - X轴: 手腕相对肩膀的前后移动 -> 机械臂X轴移动
        - Y轴: 手腕相对肩膀的左右移动 -> 机械臂Y轴移动
        - 旋转: 前臂方向角度 -> 末端执行器绕Z轴旋转
        """
        landmarks = pose_landmarks.landmark
        
        # 选择对应手臂的关键点
        if arm == 'left':
            shoulder_idx = self.keypoints['LEFT_SHOULDER']
            elbow_idx = self.keypoints['LEFT_ELBOW']
            wrist_idx = self.keypoints['LEFT_WRIST']
            calib = self.calibration_data['left']
        else:
            shoulder_idx = self.keypoints['RIGHT_SHOULDER']
            elbow_idx = self.keypoints['RIGHT_ELBOW']
            wrist_idx = self.keypoints['RIGHT_WRIST']
            calib = self.calibration_data['right']
        
        # 检查可见性
        if landmarks[wrist_idx].visibility < 0.5 or landmarks[shoulder_idx].visibility < 0.5:
            return None
        
        # 获取当前位置
        shoulder = np.array([landmarks[shoulder_idx].x, landmarks[shoulder_idx].y, landmarks[shoulder_idx].z])
        elbow = np.array([landmarks[elbow_idx].x, landmarks[elbow_idx].y, landmarks[elbow_idx].z])
        wrist = np.array([landmarks[wrist_idx].x, landmarks[wrist_idx].y, landmarks[wrist_idx].z])
        
        # 如果已校准,计算相对位移
        if calib['shoulder'] is not None and calib['reference_wrist'] is not None:
            # 手腕相对肩膀的位移
            current_relative = wrist - shoulder
            reference_relative = calib['reference_wrist'] - calib['shoulder']
            delta = current_relative - reference_relative
            
            # 映射到机器人工作空间
            # MediaPipe坐标: x(左-右), y(上-下), z(前-后)
            # 机器人坐标: X(前-后), Y(左-右), Z(上-下)
            
            # X轴: 手臂前后移动 (Z分量)
            x = np.interp(-delta[2], [-0.3, 0.3], [self.workspace_x_min, self.workspace_x_max])
            
            # Y轴: 手臂左右移动 (X分量)
            if arm == 'left':
                y = np.interp(delta[0], [-0.3, 0.3], [self.workspace_y_min, self.workspace_y_max])
            else:
                y = np.interp(-delta[0], [-0.3, 0.3], [self.workspace_y_min, self.workspace_y_max])
            
            # Z轴: 固定高度
            z = self.workspace_z
            
        else:
            # 未校准时使用默认映射
            x = np.interp(1.0 - wrist[2], [0.0, 1.0], [self.workspace_x_min, self.workspace_x_max])
            if arm == 'left':
                y = np.interp(wrist[0], [0.3, 0.7], [self.workspace_y_min, self.workspace_y_max])
            else:
                y = np.interp(wrist[0], [0.3, 0.7], [self.workspace_y_max, self.workspace_y_min])
            z = self.workspace_z
        
        # 计算旋转角度 (基于前臂方向)
        forearm_vector = wrist - elbow
        # 计算在XY平面上的投影角度
        rotation = np.arctan2(forearm_vector[1], forearm_vector[0]) * self.rotation_sensitivity
        
        position = np.array([x, y, z])
        
        return {'position': position, 'rotation': rotation}
    
    def smooth_motion(self, current, previous, alpha=None):
        """运动平滑滤波"""
        if alpha is None:
            alpha = self.smoothing
        if previous is None:
            return current
        return alpha * current + (1 - alpha) * previous
    
    def publish_endeffector_command(self, arm='left'):
        """发布末端执行器目标位置"""
        if arm == 'left':
            state = self.left_arm_state
            publisher = self.left_ee_pub
        else:
            state = self.right_arm_state
            publisher = self.right_ee_pub
        
        # 构建消息: [x, y, z, rotation_z]
        msg = Float64MultiArray()
        msg.data = [
            float(state['position'][0]),
            float(state['position'][1]),
            float(state['position'][2]),
            float(state['rotation'])
        ]
        
        publisher.publish(msg)
    
    def draw_visualization(self, image, pose_landmarks):
        """绘制可视化信息"""
        if pose_landmarks is None:
            return image
        
        landmarks = pose_landmarks.landmark
        h, w, c = image.shape
        
        # 绘制手臂骨架
        arm_connections = [
            (11, 13), (13, 15),  # 左臂
            (12, 14), (14, 16),  # 右臂
            (11, 12)  # 肩膀连线
        ]
        
        for connection in arm_connections:
            start_idx, end_idx = connection
            if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
                start_x = int(landmarks[start_idx].x * w)
                start_y = int(landmarks[start_idx].y * h)
                end_x = int(landmarks[end_idx].x * w)
                end_y = int(landmarks[end_idx].y * h)
                cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 3)
        
        # 绘制关键点
        for name, idx in self.keypoints.items():
            if landmarks[idx].visibility > 0.5:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                
                if 'SHOULDER' in name:
                    color = (255, 0, 0)  # 蓝色
                elif 'ELBOW' in name:
                    color = (0, 255, 0)  # 绿色
                else:  # WRIST
                    color = (0, 0, 255)  # 红色
                
                cv2.circle(image, (x, y), 8, color, -1)
        
        # 显示状态信息
        info_y = 30
        cv2.putText(image, f"Left EE: ({self.left_arm_state['position'][0]:.2f}, "
                   f"{self.left_arm_state['position'][1]:.2f}) Rot: {np.degrees(self.left_arm_state['rotation']):.1f}deg",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        info_y += 25
        
        cv2.putText(image, f"Right EE: ({self.right_arm_state['position'][0]:.2f}, "
                   f"{self.right_arm_state['position'][1]:.2f}) Rot: {np.degrees(self.right_arm_state['rotation']):.1f}deg",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        info_y += 25
        
        # 校准状态
        calib_status = "CALIBRATED" if self.calibration_data['left']['shoulder'] is not None else "NOT CALIBRATED"
        cv2.putText(image, f"Status: {calib_status}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if "CALIBRATED" in calib_status else (0, 0, 255), 2)
        
        # 控制说明
        cv2.putText(image, "q:Quit | c:Calibrate | r:Reset | s:Save", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def process_frame(self):
        """处理视频帧的主循环"""
        success, image = self.cap.read()
        if not success:
            return
        
        # 翻转图像
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # 姿态检测
        results = self.pose.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 处理检测结果
        if results.pose_landmarks:
            # 计算末端执行器位姿
            left_pose = self.calculate_endeffector_pose(results.pose_landmarks, 'left')
            right_pose = self.calculate_endeffector_pose(results.pose_landmarks, 'right')
            
            # 调试信息(每2秒输出一次)
            if not hasattr(self, '_last_debug_time'):
                self._last_debug_time = 0
            if time.time() - self._last_debug_time > 2.0:
                self.get_logger().info(f'检测状态: 左臂={left_pose is not None}, 右臂={right_pose is not None}')
                self._last_debug_time = time.time()
            
            # 更新左臂状态
            if left_pose is not None:
                self.left_arm_state['position'] = self.smooth_motion(
                    left_pose['position'], 
                    self.left_arm_state['prev_position']
                )
                self.left_arm_state['rotation'] = self.smooth_motion(
                    left_pose['rotation'],
                    self.left_arm_state['prev_rotation'],
                    alpha=0.5
                )
                self.left_arm_state['prev_position'] = self.left_arm_state['position'].copy()
                self.left_arm_state['prev_rotation'] = self.left_arm_state['rotation']
                
                # 发布控制命令
                self.publish_endeffector_command('left')
            else:
                if not hasattr(self, '_left_warn_time'):
                    self._left_warn_time = 0
                if time.time() - self._left_warn_time > 5.0:
                    self.get_logger().warn('左臂不可见或可见性太低')
                    self._left_warn_time = time.time()
            
            # 更新右臂状态
            if right_pose is not None:
                self.right_arm_state['position'] = self.smooth_motion(
                    right_pose['position'],
                    self.right_arm_state['prev_position']
                )
                self.right_arm_state['rotation'] = self.smooth_motion(
                    right_pose['rotation'],
                    self.right_arm_state['prev_rotation'],
                    alpha=0.5
                )
                self.right_arm_state['prev_position'] = self.right_arm_state['position'].copy()
                self.right_arm_state['prev_rotation'] = self.right_arm_state['rotation']
                
                # 发布控制命令
                self.publish_endeffector_command('right')
            else:
                if not hasattr(self, '_right_warn_time'):
                    self._right_warn_time = 0
                if time.time() - self._right_warn_time > 5.0:
                    self.get_logger().warn('右臂不可见或可见性太低')
                    self._right_warn_time = time.time()
        else:
            if not hasattr(self, '_no_pose_warn_time'):
                self._no_pose_warn_time = 0
            if time.time() - self._no_pose_warn_time > 5.0:
                self.get_logger().warn('未检测到任何人体姿态')
                self._no_pose_warn_time = time.time()
        
        # 绘制可视化
        image = self.draw_visualization(image, results.pose_landmarks)
        
        # 显示图像
        cv2.imshow('Teleoperation End-Effector Control', image)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('退出程序')
            rclpy.shutdown()
        elif key == ord('c'):
            if results.pose_landmarks:
                self.calibrate(results.pose_landmarks)
        elif key == ord('r'):
            self.calibration_data = {
                'left': {'shoulder': None, 'reference_wrist': None},
                'right': {'shoulder': None, 'reference_wrist': None}
            }
            self.get_logger().info('校准数据已重置')
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"teleop_capture_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            self.get_logger().info(f'保存截图: {filename}')
    
    def destroy_node(self):
        """清理资源"""
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TeleopEndEffectorControl()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
