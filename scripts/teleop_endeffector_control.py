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
        self.declare_parameter('workspace_x_min', 0.3)  # 工作空间X范围 (m) - 远离基座
        self.declare_parameter('workspace_x_max', 0.9)  # 扩大前方范围
        self.declare_parameter('workspace_y_min', -0.4)  # 工作空间Y范围 (m) - 扩大左右范围
        self.declare_parameter('workspace_y_max', 0.4)
        self.declare_parameter('workspace_z', 0.9)  # 固定Z高度 (m)
        self.declare_parameter('rotation_sensitivity', 2.0)  # 旋转灵敏度
        self.declare_parameter('position_smoothing', 0.15)  # 位置平滑系数（降低以减少抖动）
        self.declare_parameter('camera_id', 0)
        # 仅X/Z控制下，侧向Y固定为0（不侧偏）
        self.declare_parameter('fixed_y_left', 0.0)
        self.declare_parameter('fixed_y_right', 0.0)
        # 相对位移缩放（由图像归一化坐标映射到世界坐标范围）
        self.declare_parameter('relative_frames_for_anchor', 5)
        # 像素→米缩放配置：两种模式
        # - 'workspace': 使用工作空间范围缩放(归一化0-1乘以范围)
        # - 'fixed': 使用固定系数(米/像素)进行缩放
        self.declare_parameter('scale_mode', 'workspace')
        self.declare_parameter('scale_x_m_per_px', 0.0015)  # X每像素对应米(当scale_mode=fixed时启用)
        self.declare_parameter('scale_z_m_per_px', 0.0015)  # Z每像素对应米(当scale_mode=fixed时启用)
        self.declare_parameter('yaw_smoothing', 0.2)
        self.declare_parameter('yaw_deadband_deg', 2.0)
        # 调试：序号与同步日志
        self.declare_parameter('debug_seq_enable', True)
        self.declare_parameter('debug_log_every_n', 30)
        # 去敏感化
        self.declare_parameter('min_cam_delta', 0.02)  # 相机归一化坐标最小位移阈值
        self.declare_parameter('dominant_ratio', 1.5)  # 主臂位移相对比阈值
        self.declare_parameter('min_world_delta_x', 0.01)  # 世界坐标最小X变化(m)
        self.declare_parameter('min_world_delta_z', 0.01)  # 世界坐标最小Z变化(m)
        
        # 获取参数
        self.workspace_x_min = self.get_parameter('workspace_x_min').value
        self.workspace_x_max = self.get_parameter('workspace_x_max').value
        self.workspace_y_min = self.get_parameter('workspace_y_min').value
        self.workspace_y_max = self.get_parameter('workspace_y_max').value
        self.workspace_z = self.get_parameter('workspace_z').value
        self.rotation_sensitivity = self.get_parameter('rotation_sensitivity').value
        self.smoothing = self.get_parameter('position_smoothing').value
        camera_id = self.get_parameter('camera_id').value
        self.fixed_y_left = float(self.get_parameter('fixed_y_left').value)
        self.fixed_y_right = float(self.get_parameter('fixed_y_right').value)
        self.anchor_wait_frames = int(self.get_parameter('relative_frames_for_anchor').value)
        self.scale_mode = str(self.get_parameter('scale_mode').value)
        self.scale_x_m_per_px = float(self.get_parameter('scale_x_m_per_px').value)
        self.scale_z_m_per_px = float(self.get_parameter('scale_z_m_per_px').value)
        self.yaw_alpha = float(self.get_parameter('yaw_smoothing').value)
        self.yaw_deadband = np.deg2rad(float(self.get_parameter('yaw_deadband_deg').value))
        self.min_cam_delta = float(self.get_parameter('min_cam_delta').value)
        self.dominant_ratio = float(self.get_parameter('dominant_ratio').value)
        self.min_world_delta_x = float(self.get_parameter('min_world_delta_x').value)
        self.min_world_delta_z = float(self.get_parameter('min_world_delta_z').value)
        self.debug_seq_enable = bool(self.get_parameter('debug_seq_enable').value)
        self.debug_log_every_n = int(self.get_parameter('debug_log_every_n').value)
        
        # 添加夹爪控制发布器
        self.left_gripper_pub = self.create_publisher(
            Float64MultiArray,
            '/left_arm/gripper_command',
            10
        )
        self.right_gripper_pub = self.create_publisher(
            Float64MultiArray,
            '/right_arm/gripper_command',
            10
        )
        
        # 初始化MediaPipe Pose和Hands
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 姿态关键点索引
        self.keypoints = {
            'LEFT_SHOULDER': 11,
            'LEFT_ELBOW': 13,
            'LEFT_WRIST': 15,
            'RIGHT_SHOULDER': 12,
            'RIGHT_ELBOW': 14,
            'RIGHT_WRIST': 16,
        }
        
        # Z轴控制范围 (根据手腕高度)
        self.declare_parameter('workspace_z_min', 0.5)  # 最低高度
        self.declare_parameter('workspace_z_max', 1.2)  # 最高高度
        self.workspace_z_min = self.get_parameter('workspace_z_min').value
        self.workspace_z_max = self.get_parameter('workspace_z_max').value
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 {camera_id}')
            raise RuntimeError('摄像头初始化失败')
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 定义初始位置（用于重置）
        self.INITIAL_LEFT_POSITION = np.array([0.6, 0.3, 0.7])
        self.INITIAL_RIGHT_POSITION = np.array([0.6, -0.3, 0.7])
        
        # 状态变量 - 机械臂状态
        self.left_arm_state = {
            'position': self.INITIAL_LEFT_POSITION.copy(),  # 左臂初始位置(绝对)
            'prev_position': None,
            'yaw': 0.0,
            'prev_yaw': None,
            'gripper_open': True,  # 夹爪状态
            'prev_gripper_open': True,
            # 相对位移锚点与当前相对量（基于图像手腕点）
            'anchor_cam_xy': None,
            'anchor_counter': 0,
            'relative_world_dx': 0.0,
            'relative_world_dz': 0.0,
            # 最近一次可用于日志的相对目标（dx, dz, yaw）
            'last_relative': (0.0, 0.0, 0.0)
        }
        self.right_arm_state = {
            'position': self.INITIAL_RIGHT_POSITION.copy(),  # 右臂初始位置(绝对)
            'prev_position': None,
            'yaw': 0.0,
            'prev_yaw': None,
            'gripper_open': True,
            'prev_gripper_open': True,
            'anchor_cam_xy': None,
            'anchor_counter': 0,
            'relative_world_dx': 0.0,
            'relative_world_dz': 0.0,
            'last_relative': (0.0, 0.0, 0.0)
        }
        
        # 手势识别状态（用于去抖动）
        self.gesture_buffer = {
            'left': [],  # 存储最近N帧的手势
            'right': []
        }
        self.gesture_buffer_size = 5  # 缓冲区大小
        
        # 手臂可见性去抖动 - 必须连续N帧不可见才停止发送命令
        self.visibility_buffer = {
            'left': [],   # 存储可见性状态
            'right': []
        }
        self.visibility_buffer_size = 5  # 需要连续5帧判断
        self.visibility_threshold = 0.8  # 需要80%的帧可见才认为稳定可见
        
        # 创建定时器处理视频流
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30Hz
        
        self.get_logger().info('末端执行器遥操作控制节点已启动')
        self.get_logger().info(f'工作空间: X[{self.workspace_x_min}, {self.workspace_x_max}], '
                              f'Y[{self.workspace_y_min}, {self.workspace_y_max}], '
                              f'Z[{self.workspace_z_min}, {self.workspace_z_max}]')
        self.get_logger().info('=== 使用说明 ===')
        self.get_logger().info('手臂控制: 3DOF(X/Z/偏航) + 手势')
        self.get_logger().info('映射: 手腕X→末端X, 手腕高度Y→末端Z, 手腕旋转→末端偏航')
        self.get_logger().info('  - 手臂左右移动 → 机械臂平面内左右移动')
        self.get_logger().info('  - 抬起手臂 → 机械臂上升')
        self.get_logger().info('  - 掌心面向屏幕旋转 → 机械臂旋转手腕关节')
        self.get_logger().info('手势控制:')
        self.get_logger().info('  - 张开手掌 → 打开夹爪')
        self.get_logger().info('  - 握拳 → 关闭夹爪')
        self.get_logger().info('按键: q-退出 | r-重置到初始位置 | s-保存截图')
        if self.debug_seq_enable:
            self.get_logger().info(f'调试序号开启：每 {self.debug_log_every_n} 帧打印一次（基于 seq）')
        else:
            self.get_logger().info('调试序号关闭：按内部计数打印日志')
    
    def _get_pose_wrist_pixels(self, pose_landmarks, image_shape):
        """返回可见的左右手腕像素坐标: {'left': (x,y) or None, 'right': (x,y) or None}"""
        if pose_landmarks is None:
            return {'left': None, 'right': None}
        h, w, _ = image_shape
        # 注意：为与calculate_endeffector_pose中的镜像策略保持一致，
        # 这里将“left臂”的参考腕像素取自RIGHT_WRIST，“right臂”的参考腕像素取自LEFT_WRIST。
        l_idx = self.keypoints['RIGHT_WRIST']  # 对应left臂
        r_idx = self.keypoints['LEFT_WRIST']   # 对应right臂
        lm = pose_landmarks.landmark
        left = None
        right = None
        if lm[l_idx].visibility > 0.5:
            left = (float(lm[l_idx].x * w), float(lm[l_idx].y * h))
        if lm[r_idx].visibility > 0.5:
            right = (float(lm[r_idx].x * w), float(lm[r_idx].y * h))
        return {'left': left, 'right': right}

    def _assign_arm_for_hand(self, hand_landmarks, wrists_px, image_shape, fallback_label):
        """基于最近手腕分配左右臂；若两腕都不可用则回退到handedness标签。"""
        if hand_landmarks is None:
            return 'left' if fallback_label == 'Left' else 'right'
        h, w, _ = image_shape
        hx = float(hand_landmarks.landmark[0].x * w)
        hy = float(hand_landmarks.landmark[0].y * h)
        left_px = wrists_px.get('left')
        right_px = wrists_px.get('right')
        # 两腕都不可见，回退到handedness
        if left_px is None and right_px is None:
            return 'left' if fallback_label == 'Left' else 'right'
        # 仅一侧可见
        if right_px is None:
            return 'left'
        if left_px is None:
            return 'right'
        # 比较距离，选最近腕
        dl = (hx - left_px[0])**2 + (hy - left_px[1])**2
        dr = (hx - right_px[0])**2 + (hy - right_px[1])**2
        return 'left' if dl <= dr else 'right'

    def reset_to_initial_position(self):
        """重置机械臂到初始位置"""
        self.get_logger().info('重置机械臂到初始位置...')
        
        # 重置左臂
        self.left_arm_state['position'] = self.INITIAL_LEFT_POSITION.copy()
        self.left_arm_state['prev_position'] = None
        self.left_arm_state['gripper_open'] = True
        # 清锚点/相对
        self.left_arm_state['anchor_cam_xy'] = None
        self.left_arm_state['anchor_counter'] = 0
        self.left_arm_state['relative_world_dx'] = 0.0
        self.left_arm_state['relative_world_dz'] = 0.0
        self.left_arm_state['last_relative'] = (0.0, 0.0, 0.0)
        self.left_arm_state['yaw'] = 0.0
        self.left_arm_state['prev_yaw'] = None

        self.publish_endeffector_command('left')
        self.publish_gripper_command('left', True)
        
        # 重置右臂
        self.right_arm_state['position'] = self.INITIAL_RIGHT_POSITION.copy()
        self.right_arm_state['prev_position'] = None
        self.right_arm_state['gripper_open'] = True
        # 清锚点/相对
        self.right_arm_state['anchor_cam_xy'] = None
        self.right_arm_state['anchor_counter'] = 0
        self.right_arm_state['relative_world_dx'] = 0.0
        self.right_arm_state['relative_world_dz'] = 0.0
        self.right_arm_state['last_relative'] = (0.0, 0.0, 0.0)
        self.right_arm_state['yaw'] = 0.0
        self.right_arm_state['prev_yaw'] = None

        self.publish_endeffector_command('right')
        self.publish_gripper_command('right', True)
        
        # 清空手势缓冲
        self.gesture_buffer['left'].clear()
        self.gesture_buffer['right'].clear()
        
        self.get_logger().info('✅ 已重置到初始位置: 左臂(0.6, 0.3, 0.7), 右臂(0.6, -0.3, 0.7)')

    def estimate_wrist_yaw(self, hand_landmarks):
        """根据手部关键点估计手腕绕垂直轴的偏航角(弧度)，精度可较低，附带去噪"""
        if hand_landmarks is None:
            return None
        lm = hand_landmarks.landmark
        # 选用腕点(0)到中指根(9)和食指根(5)的向量平均作为手掌朝向
        try:
            wx, wy = lm[0].x, lm[0].y
            v1 = np.array([lm[9].x - wx, lm[9].y - wy])
            v2 = np.array([lm[5].x - wx, lm[5].y - wy])
            v = (v1 + v2) * 0.5
            if np.linalg.norm(v) < 1e-6:
                return None
            # 图像坐标系: x向右, y向下。atan2(dy,dx)
            raw_angle = np.arctan2(v[1], v[0])
            # 将角度映射到[-pi,pi]
            if raw_angle > np.pi:
                raw_angle -= 2*np.pi
            if raw_angle < -np.pi:
                raw_angle += 2*np.pi
            return float(raw_angle)
        except Exception:
            return None

    def smooth_yaw(self, new_yaw, prev_yaw):
        """对偏航进行一阶低通与小死区处理"""
        if new_yaw is None:
            return prev_yaw if prev_yaw is not None else 0.0
        if prev_yaw is None:
            prev_yaw = new_yaw
        # 小死区: 角度变化过小则忽略
        if abs(new_yaw - prev_yaw) < self.yaw_deadband:
            return prev_yaw
        return self.yaw_alpha * new_yaw + (1.0 - self.yaw_alpha) * prev_yaw

    def update_relative_from_cam(self, arm_state, cam_xy):
        """
        基于图像手腕点计算相对位移：
        - 初始化点 a：首次检测到该臂手腕点，记录为 anchor，并开始计数；达到 anchor_wait_frames 认为稳定
        - 位置 b-a：当前手腕点与 anchor 之差，映射到世界相对位移(dx,dz)
        - 缩放策略：
          * scale_mode='workspace'：使用归一化差值×(X/Z工作空间范围)
          * scale_mode='fixed'：将像素差值×(scale_x_m_per_px / scale_z_m_per_px)
        - Z方向：图像y越小越高，因此 dz 为 -(cy - ay) × 缩放
        - 去敏感化：相机最小位移阈值(min_cam_delta)、世界最小变化阈值(min_world_delta_x/z)
        - 记录 last_relative=(dx,dz,yaw) 供日志打印
        """
        if cam_xy is None:
            return
        if arm_state['anchor_cam_xy'] is None:
            arm_state['anchor_cam_xy'] = cam_xy
            arm_state['anchor_counter'] = 1
            arm_state['relative_world_dx'] = 0.0
            arm_state['relative_world_dz'] = 0.0
            arm_state['last_relative'] = (0.0, 0.0, float(arm_state['yaw'] if arm_state['yaw'] is not None else 0.0))
            return
        # 锚定稳定期计数
        if arm_state['anchor_counter'] < self.anchor_wait_frames:
            arm_state['anchor_counter'] += 1
        # 计算相对(相对于首次可见锚点a)，图像坐标差->世界坐标差
        ax, ay = arm_state['anchor_cam_xy']
        cx, cy = cam_xy
        # 去敏感化：若相机位移过小，保持相对位移不变
        cam_dx = cx - ax
        cam_dy = cy - ay
        cam_dist = np.hypot(cam_dx, cam_dy)
        if cam_dist < self.min_cam_delta:
            dx_world = arm_state['relative_world_dx']
            dz_world = arm_state['relative_world_dz']
        else:
            if self.scale_mode == 'fixed':
                # 使用像素差值与固定缩放系数
                dx_world = cam_dx * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_x_m_per_px
                dz_world = -cam_dy * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_z_m_per_px
            else:
                # 使用归一化差值与工作空间范围
                dx_world = cam_dx * (self.workspace_x_max - self.workspace_x_min)
                dz_world = -cam_dy * (self.workspace_z_max - self.workspace_z_min)
            # 世界层面的死区
            if abs(dx_world - arm_state['relative_world_dx']) < self.min_world_delta_x:
                dx_world = arm_state['relative_world_dx']
            if abs(dz_world - arm_state['relative_world_dz']) < self.min_world_delta_z:
                dz_world = arm_state['relative_world_dz']
        arm_state['relative_world_dx'] = float(dx_world)
        arm_state['relative_world_dz'] = float(dz_world)
        arm_state['last_relative'] = (
            float(dx_world),
            float(dz_world),
            float(arm_state['yaw'] if arm_state['yaw'] is not None else 0.0)
        )
    
    def recognize_hand_gesture(self, hand_landmarks):
        """识别手势：返回'open'(张开)或'closed'(握拳)"""
        if hand_landmarks is None:
            return None
        
        landmarks = hand_landmarks.landmark
        
        # 计算手指伸展程度
        # 使用指尖和手掌的距离判断
        palm_center = landmarks[0]  # 手腕
        
        # 检查四个手指（食指、中指、无名指、小指）是否伸直
        finger_tips = [8, 12, 16, 20]  # 指尖
        finger_mcp = [5, 9, 13, 17]   # 手指根部
        
        extended_fingers = 0
        for tip_idx, mcp_idx in zip(finger_tips, finger_mcp):
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            palm = palm_center
            
            # 计算距离
            tip_dist = np.sqrt((tip.x - palm.x)**2 + (tip.y - palm.y)**2)
            mcp_dist = np.sqrt((mcp.x - palm.x)**2 + (mcp.y - palm.y)**2)
            
            # 如果指尖距离明显大于根部距离，认为手指伸直
            if tip_dist > mcp_dist * 1.3:
                extended_fingers += 1
        
        # 判断手势
        if extended_fingers >= 3:  # 至少3个手指伸直 → 手掌张开
            return 'open'
        elif extended_fingers <= 1:  # 最多1个手指伸直 → 握拳
            return 'closed'
        else:
            return None  # 中间状态，不确定
    
    def update_gesture_buffer(self, arm, gesture):
        """更新手势缓冲区并返回稳定的手势（去抖动）"""
        buffer = self.gesture_buffer[arm]
        
        if gesture is not None:
            buffer.append(gesture)
        
        # 保持缓冲区大小
        if len(buffer) > self.gesture_buffer_size:
            buffer.pop(0)
        
        # 如果缓冲区中大多数是同一手势，返回该手势
        if len(buffer) >= self.gesture_buffer_size:
            open_count = buffer.count('open')
            closed_count = buffer.count('closed')
            
            if open_count > self.gesture_buffer_size * 0.7:
                return True  # 张开
            elif closed_count > self.gesture_buffer_size * 0.7:
                return False  # 握拳
        
        return None  # 不确定
    
    def calculate_endeffector_pose(self, pose_landmarks, arm='left'):
        """
        计算末端执行器位姿 (3D位置 - X, Y, Z) 及图像手腕点

        新映射策略(按需求):
        - X轴: 手腕X图像坐标 → 机械臂X轴(前后)
        - Z轴: 手腕Y图像高度 → 机械臂Z轴(上下)
        - Y轴: 固定到该臂侧的初始偏移
        """
        landmarks = pose_landmarks.landmark
        
        # 选择对应手臂的关键点（镜像：left臂使用右侧关键点，right臂使用左侧关键点）
        if arm == 'left':
            shoulder_idx = self.keypoints['RIGHT_SHOULDER']
            wrist_idx = self.keypoints['RIGHT_WRIST']
        else:
            shoulder_idx = self.keypoints['LEFT_SHOULDER']
            wrist_idx = self.keypoints['LEFT_WRIST']
        
        # 检查可见性（提高阈值，更严格）
        if landmarks[wrist_idx].visibility < 0.7 or landmarks[shoulder_idx].visibility < 0.7:
            return None
        
        # 获取图像坐标（0-1）
        wrist = np.array([landmarks[wrist_idx].x, landmarks[wrist_idx].y])
        # 合理范围校验
        if not (0.05 < wrist[0] < 0.95 and 0.05 < wrist[1] < 0.95):
            return None

        # X: 直接用图像x从[0,1]映射到工作空间[x_min,x_max]
        x = np.interp(wrist[0], [0.0, 1.0], [self.workspace_x_min, self.workspace_x_max])
        # Z: 图像y越小越高，映射到[z_min,z_max]
        z = np.interp(1.0 - wrist[1], [0.0, 1.0], [self.workspace_z_min, self.workspace_z_max])
        # Y固定
        y = self.fixed_y_left if arm == 'left' else self.fixed_y_right
        
        # 限制在工作空间内
        x = np.clip(x, self.workspace_x_min, self.workspace_x_max)
        y = np.clip(y, self.workspace_y_min, self.workspace_y_max)
        z = np.clip(z, self.workspace_z_min, self.workspace_z_max)
        
        position = np.array([x, y, z])
        # 返回图像手腕点用于相对位移计算
        return {'position': position, 'cam_wrist': (float(wrist[0]), float(wrist[1]))}
    
    def smooth_motion(self, current, previous, alpha=None):
        """运动平滑滤波"""
        if alpha is None:
            alpha = self.smoothing
        if previous is None:
            return current
        return alpha * current + (1 - alpha) * previous
    
    def publish_endeffector_command(self, arm='left'):
        """发布相对末端执行器目标: [ΔX, ΔZ, ΔYaw, (可选)seq] 并输出日志"""
        # 常规发布：left臂→left话题，right臂→right话题
        if arm == 'left':
            state = self.left_arm_state
            publisher = self.left_ee_pub
        else:
            state = self.right_arm_state
            publisher = self.right_ee_pub
        
        # 构建消息: [ΔX, ΔZ, ΔYaw, (可选)seq]
        rel_dx, rel_dz, rel_yaw = state.get('last_relative', (0.0, 0.0, 0.0))
        msg = Float64MultiArray()
        payload = [float(rel_dx), float(rel_dz), float(rel_yaw)]
        if self.debug_seq_enable:
            if not hasattr(self, f'_{arm}_seq'):
                setattr(self, f'_{arm}_seq', 0)
            seq = getattr(self, f'_{arm}_seq') + 1
            setattr(self, f'_{arm}_seq', seq)
            payload.append(float(seq))
        msg.data = payload
        
        publisher.publish(msg)
        
        # 调试信息：定期输出发布的目标位置
        if self.debug_seq_enable:
            seq = int(payload[3]) if len(payload) > 3 else -1
            if seq > 0 and seq % self.debug_log_every_n == 0:
                self.get_logger().info(f'[{arm}臂] 发送相对位置目标xyz: x={rel_dx:.3f}m, y={rel_dz:.3f}m, z={rel_yaw:.2f}rad (seq={seq})')
        else:
            if not hasattr(self, f'_{arm}_pub_count'):
                setattr(self, f'_{arm}_pub_count', 0)
            count = getattr(self, f'_{arm}_pub_count') + 1
            setattr(self, f'_{arm}_pub_count', count)
            if count % self.debug_log_every_n == 0:
                self.get_logger().info(f'[{arm}臂] 发送相对位置目标xyz: x={rel_dx:.3f}m, y={rel_dz:.3f}m, z={rel_yaw:.2f}rad')
    
    def publish_gripper_command(self, arm='left', open_gripper=True):
        """发布夹爪命令"""
        # 常规发布：left臂→left夹爪话题，right臂→right夹爪话题
        if arm == 'left':
            publisher = self.left_gripper_pub
            state = self.left_arm_state
        else:
            publisher = self.right_gripper_pub
            state = self.right_arm_state
        
        # 避免重复发送相同命令
        if state['prev_gripper_open'] == open_gripper:
            return
        
        # 构建消息: [gripper_position] (0.0=关闭, 1.0=打开)
        msg = Float64MultiArray()
        msg.data = [1.0 if open_gripper else 0.0]
        
        publisher.publish(msg)
        state['prev_gripper_open'] = open_gripper
        
        self.get_logger().info(f'{arm}夹爪: {"打开" if open_gripper else "关闭"}')
    
    def draw_visualization(self, image, pose_landmarks, hands_results):
        """绘制可视化信息"""
        h, w, c = image.shape
        
        # 绘制姿态骨架
        if pose_landmarks is not None:
            landmarks = pose_landmarks.landmark
            
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
        
        # 绘制手部关键点
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        
        # 显示状态信息
        info_y = 30
        left_gripper_status = "打开" if self.left_arm_state['gripper_open'] else "关闭"
        cv2.putText(image, f"Left: ({self.left_arm_state['position'][0]:.2f}, "
                   f"{self.left_arm_state['position'][1]:.2f}, {self.left_arm_state['position'][2]:.2f}) "
                   f"Gripper: {left_gripper_status}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        info_y += 25
        
        right_gripper_status = "打开" if self.right_arm_state['gripper_open'] else "关闭"
        cv2.putText(image, f"Right: ({self.right_arm_state['position'][0]:.2f}, "
                   f"{self.right_arm_state['position'][1]:.2f}, {self.right_arm_state['position'][2]:.2f}) "
                   f"Gripper: {right_gripper_status}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        info_y += 25
        
        # 控制说明
        cv2.putText(image, "q:Quit | r:Reset to Initial | s:Save | Open Palm=Open | Fist=Close", 
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
        
        # 姿态和手部检测
        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 处理姿态检测结果
        if pose_results.pose_landmarks:
            # 计算末端执行器位姿
            left_pose = self.calculate_endeffector_pose(pose_results.pose_landmarks, 'left')
            right_pose = self.calculate_endeffector_pose(pose_results.pose_landmarks, 'right')
            
            # 更新可见性缓冲区（去抖动）
            self.visibility_buffer['left'].append(left_pose is not None)
            self.visibility_buffer['right'].append(right_pose is not None)
            
            # 保持缓冲区大小
            if len(self.visibility_buffer['left']) > self.visibility_buffer_size:
                self.visibility_buffer['left'].pop(0)
            if len(self.visibility_buffer['right']) > self.visibility_buffer_size:
                self.visibility_buffer['right'].pop(0)
            
            # 判断是否稳定可见：需要达到阈值比例才认为可见
            left_is_visible = sum(self.visibility_buffer['left']) >= len(self.visibility_buffer['left']) * self.visibility_threshold
            right_is_visible = sum(self.visibility_buffer['right']) >= len(self.visibility_buffer['right']) * self.visibility_threshold
            
            # 更新左臂位置（加入去敏感化）
            if left_pose is not None and left_is_visible:
                self.left_arm_state['position'] = self.smooth_motion(
                    left_pose['position'], 
                    self.left_arm_state['prev_position']
                )
                self.left_arm_state['prev_position'] = self.left_arm_state['position'].copy()
                # 更新相对位移(基于图像手腕点)
                self.update_relative_from_cam(self.left_arm_state, left_pose.get('cam_wrist'))
                
                # 发布控制命令
                self.publish_endeffector_command('left')
                
                # 重置警告标志并通知恢复
                if hasattr(self, '_left_invisible_warned'):
                    del self._left_invisible_warned
                    self.get_logger().info('✅ 左臂重新可见，继续跟随')
            else:
                # 手臂不可见时，保持当前位置，不更新也不发布命令
                if not hasattr(self, '_left_invisible_warned'):
                    self._left_invisible_warned = True
                    self.get_logger().info('⚠️  左臂不可见，保持当前位置不动')
            
            # 更新右臂位置（加入去敏感化）
            if right_pose is not None and right_is_visible:
                self.right_arm_state['position'] = self.smooth_motion(
                    right_pose['position'],
                    self.right_arm_state['prev_position']
                )
                self.right_arm_state['prev_position'] = self.right_arm_state['position'].copy()
                self.update_relative_from_cam(self.right_arm_state, right_pose.get('cam_wrist'))
                
                # 发布控制命令
                self.publish_endeffector_command('right')
                
                # 重置警告标志并通知恢复
                if hasattr(self, '_right_invisible_warned'):
                    del self._right_invisible_warned
                    self.get_logger().info('✅ 右臂重新可见，继续跟随')
            else:
                # 手臂不可见时，保持当前位置，不更新也不发布命令
                if not hasattr(self, '_right_invisible_warned'):
                    self._right_invisible_warned = True
                    self.get_logger().info('⚠️  右臂不可见，保持当前位置不动')
        else:
            # 完全没有检测到人体姿态
            if not hasattr(self, '_no_pose_warned'):
                self._no_pose_warned = True
                self.get_logger().debug('⚠️ 未检测到人体姿态，机械臂保持当前位置')
        
        # 处理手部手势识别 + 偏航估计
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            wrists_px = self._get_pose_wrist_pixels(pose_results.pose_landmarks if pose_results else None, image.shape)
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                # 确定是左手还是右手（注意镜像翻转）
                hand_label = handedness.classification[0].label
                # 绑定规则：优先使用最近手腕分配；若不可判定则回退到MediaPipe标签
                arm = self._assign_arm_for_hand(hand_landmarks, wrists_px, image.shape, hand_label)
                
                # 识别手势
                gesture = self.recognize_hand_gesture(hand_landmarks)
                stable_gesture = self.update_gesture_buffer(arm, gesture)
                
                # 根据稳定的手势更新夹爪状态
                if stable_gesture is not None:
                    if arm == 'left':
                        if stable_gesture != self.left_arm_state['gripper_open']:
                            self.left_arm_state['gripper_open'] = stable_gesture
                            self.publish_gripper_command('left', stable_gesture)
                    else:
                        if stable_gesture != self.right_arm_state['gripper_open']:
                            self.right_arm_state['gripper_open'] = stable_gesture
                            self.publish_gripper_command('right', stable_gesture)

                # 估计偏航角并平滑
                yaw_new = self.estimate_wrist_yaw(hand_landmarks)
                if arm == 'left':
                    self.left_arm_state['yaw'] = self.smooth_yaw(yaw_new, self.left_arm_state['prev_yaw'])
                    self.left_arm_state['prev_yaw'] = self.left_arm_state['yaw']
                    # 更新last_relative中的yaw部分
                    dx, dz, _ = self.left_arm_state['last_relative']
                    self.left_arm_state['last_relative'] = (dx, dz, float(self.left_arm_state['yaw']))
                else:
                    self.right_arm_state['yaw'] = self.smooth_yaw(yaw_new, self.right_arm_state['prev_yaw'])
                    self.right_arm_state['prev_yaw'] = self.right_arm_state['yaw']
                    dx, dz, _ = self.right_arm_state['last_relative']
                    self.right_arm_state['last_relative'] = (dx, dz, float(self.right_arm_state['yaw']))
        
        # 绘制可视化
        image = self.draw_visualization(image, pose_results.pose_landmarks, hands_results)
        
        # 显示图像
        cv2.imshow('Teleoperation End-Effector Control', image)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('退出程序')
            rclpy.shutdown()
        elif key == ord('r'):
            # 重置到初始位置
            self.reset_to_initial_position()
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
