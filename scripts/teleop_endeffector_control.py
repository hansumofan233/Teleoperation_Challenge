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
            'position': self.INITIAL_LEFT_POSITION.copy(),  # 左臂初始位置
            'prev_position': None,
            'gripper_open': True,  # 夹爪状态
            'prev_gripper_open': True
        }
        self.right_arm_state = {
            'position': self.INITIAL_RIGHT_POSITION.copy(),  # 右臂初始位置
            'prev_position': None,
            'gripper_open': True,
            'prev_gripper_open': True
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
        self.visibility_buffer_size = 3  # 需要连续3帧不可见才停止
        
        # 创建定时器处理视频流
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30Hz
        
        self.get_logger().info('末端执行器遥操作控制节点已启动')
        self.get_logger().info(f'工作空间: X[{self.workspace_x_min}, {self.workspace_x_max}], '
                              f'Y[{self.workspace_y_min}, {self.workspace_y_max}], '
                              f'Z[{self.workspace_z_min}, {self.workspace_z_max}]')
        self.get_logger().info('=== 使用说明 ===')
        self.get_logger().info('手臂控制: 直接映射模式')
        self.get_logger().info('  - 向前伸手 → 机械臂向前移动')
        self.get_logger().info('  - 左右移动 → 机械臂左右移动')
        self.get_logger().info('  - 抬起手臂 → 机械臂上升')
        self.get_logger().info('手势控制:')
        self.get_logger().info('  - 张开手掌 → 打开夹爪')
        self.get_logger().info('  - 握拳 → 关闭夹爪')
        self.get_logger().info('按键: q-退出 | r-重置到初始位置 | s-保存截图')
    
    def reset_to_initial_position(self):
        """重置机械臂到初始位置"""
        self.get_logger().info('重置机械臂到初始位置...')
        
        # 重置左臂
        self.left_arm_state['position'] = self.INITIAL_LEFT_POSITION.copy()
        self.left_arm_state['prev_position'] = None
        self.left_arm_state['gripper_open'] = True
        self.publish_endeffector_command('left')
        self.publish_gripper_command('left', True)
        
        # 重置右臂
        self.right_arm_state['position'] = self.INITIAL_RIGHT_POSITION.copy()
        self.right_arm_state['prev_position'] = None
        self.right_arm_state['gripper_open'] = True
        self.publish_endeffector_command('right')
        self.publish_gripper_command('right', True)
        
        # 清空手势缓冲
        self.gesture_buffer['left'].clear()
        self.gesture_buffer['right'].clear()
        
        self.get_logger().info('✅ 已重置到初始位置: 左臂(0.6, 0.3, 0.7), 右臂(0.6, -0.3, 0.7)')
    
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
        计算末端执行器位姿 (3D位置 - X, Y, Z)
        
        新映射策略:
        - X轴: 手腕前后位置 -> 机械臂X轴移动
        - Y轴: 手腕左右位置 -> 机械臂Y轴移动
        - Z轴: 手腕高度 -> 机械臂Z轴移动（抬手=上升）
        """
        landmarks = pose_landmarks.landmark
        
        # 选择对应手臂的关键点
        if arm == 'left':
            shoulder_idx = self.keypoints['LEFT_SHOULDER']
            wrist_idx = self.keypoints['LEFT_WRIST']
        else:
            shoulder_idx = self.keypoints['RIGHT_SHOULDER']
            wrist_idx = self.keypoints['RIGHT_WRIST']
        
        # 检查可见性（提高阈值，更严格）
        if landmarks[wrist_idx].visibility < 0.7 or landmarks[shoulder_idx].visibility < 0.7:
            return None
        
        # 获取当前位置
        shoulder = np.array([landmarks[shoulder_idx].x, landmarks[shoulder_idx].y, landmarks[shoulder_idx].z])
        wrist = np.array([landmarks[wrist_idx].x, landmarks[wrist_idx].y, landmarks[wrist_idx].z])
        
        # 检查手腕是否在合理范围内（防止部分遮挡或边缘情况）
        # MediaPipe坐标范围是[0, 1]，我们要求在[0.1, 0.9]范围内
        if not (0.1 < wrist[0] < 0.9 and 0.1 < wrist[1] < 0.9):
            return None  # 手腕太靠近画面边缘，可能即将离开
        
        # 使用相对于肩膀的位置进行映射
        relative = wrist - shoulder
        
        # X轴: 手臂前后伸展
        # MediaPipe Z: 负值=向前，正值=向后
        x = np.interp(-relative[2], [-0.5, 0.3], [self.workspace_x_min, self.workspace_x_max])
        
        # Y轴: 手臂左右位置
        if arm == 'left':
            # 左臂：手向左移动→机械臂向左移动
            y = np.interp(relative[0], [-0.2, 0.5], [self.workspace_y_min, self.workspace_y_max])
        else:
            # 右臂：手向右移动→机械臂向右移动
            y = np.interp(-relative[0], [-0.2, 0.5], [self.workspace_y_min, self.workspace_y_max])
        
        # Z轴: 手腕高度（抬手=上升）
        # MediaPipe Y: 0(上) 到 1(下)，需要反转
        # 使用手腕的绝对Y坐标
        z = np.interp(1.0 - wrist[1], [0.3, 0.9], [self.workspace_z_min, self.workspace_z_max])
        
        # 限制在工作空间内
        x = np.clip(x, self.workspace_x_min, self.workspace_x_max)
        y = np.clip(y, self.workspace_y_min, self.workspace_y_max)
        z = np.clip(z, self.workspace_z_min, self.workspace_z_max)
        
        position = np.array([x, y, z])
        
        return {'position': position}
    
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
        
        # 构建消息: [x, y, z]
        msg = Float64MultiArray()
        msg.data = [
            float(state['position'][0]),
            float(state['position'][1]),
            float(state['position'][2])
        ]
        
        publisher.publish(msg)
    
    def publish_gripper_command(self, arm='left', open_gripper=True):
        """发布夹爪命令"""
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
            
            # 判断是否稳定可见：缓冲区中至少有一半是可见的
            left_is_visible = sum(self.visibility_buffer['left']) >= len(self.visibility_buffer['left']) * 0.5
            right_is_visible = sum(self.visibility_buffer['right']) >= len(self.visibility_buffer['right']) * 0.5
            
            # 更新左臂位置（只有当稳定可见且有有效位姿时才更新）
            if left_pose is not None and left_is_visible:
                self.left_arm_state['position'] = self.smooth_motion(
                    left_pose['position'], 
                    self.left_arm_state['prev_position']
                )
                self.left_arm_state['prev_position'] = self.left_arm_state['position'].copy()
                
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
            
            # 更新右臂位置（只有当稳定可见且有有效位姿时才更新）
            if right_pose is not None and right_is_visible:
                self.right_arm_state['position'] = self.smooth_motion(
                    right_pose['position'],
                    self.right_arm_state['prev_position']
                )
                self.right_arm_state['prev_position'] = self.right_arm_state['position'].copy()
                
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
                self.get_logger().debug('未检测到人体姿态，机械臂保持当前位置')
        
        # 处理手部手势识别
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                # 确定是左手还是右手（注意镜像翻转）
                hand_label = handedness.classification[0].label
                
                # MediaPipe的Left/Right是从摄像头视角镜像翻转
                if hand_label == "Left":
                    arm = "left"   # MediaPipe的Left = 用户左手 = 左臂
                else:
                    arm = "right"  # MediaPipe的Right = 用户右手 = 右臂
                
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
