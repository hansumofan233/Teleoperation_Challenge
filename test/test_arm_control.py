#!/usr/bin/env python3
"""
定制的机器人手臂控制检测脚本
只检测和显示肩、肘、腕、手部关键点
专门为UR3e双臂机器人设计
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys

class ArmControlDetection:
    def __init__(self):
        print("正在初始化定制手臂检测系统...")
        
        # 初始化MediaPipe姿态检测
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # 初始化MediaPipe手部检测（获得更精确的手部信息）
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        
        # 定义机器人控制需要的关键点（仅手臂部分）
        self.arm_keypoints = {
            # 左臂
            'LEFT_SHOULDER': 11,
            'LEFT_ELBOW': 13,
            'LEFT_WRIST': 15,
            # 右臂
            'RIGHT_SHOULDER': 12,
            'RIGHT_ELBOW': 14,
            'RIGHT_WRIST': 16,
        }
        
        # 定义手臂连接（仅绘制手臂骨架）
        self.arm_connections = [
            (11, 13),  # 左肩到左肘
            (13, 15),  # 左肘到左腕
            (12, 14),  # 右肩到右肘  
            (14, 16),  # 右肘到右腕
            (11, 12),  # 左肩到右肩
        ]
        
        # 初始化摄像头
        print("正在连接摄像头...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("摄像头0无法打开，尝试摄像头1...")
            self.cap = cv2.VideoCapture(1)
            
        if not self.cap.isOpened():
            print("错误：无法打开任何摄像头")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ 定制手臂检测系统启动成功")
        print("✓ 摄像头连接成功")
        print("控制说明：")
        print("  - 按 'q' 键退出")
        print("  - 按 's' 键保存当前帧")
        print("  - 按 'h' 切换手部细节显示")
        print("  - 蓝色：肩膀关节")
        print("  - 绿色：肘部关节") 
        print("  - 红色：手腕关节")
        print("  - 黄色：手部关键点")

    def calculate_joint_angles(self, pose_landmarks):
        """计算机器人控制需要的关节角度"""
        def get_angle_3d(p1, p2, p3):
            """计算3D空间中三点的角度"""
            v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        angles = {}
        landmarks = pose_landmarks.landmark
        
        # 左臂关节角度
        if all(landmarks[i].visibility > 0.5 for i in [11, 13, 15]):
            # 左肩角度 (上臂与躯干的角度)
            angles['left_shoulder'] = get_angle_3d(
                landmarks[13], landmarks[11], landmarks[12]  # 肘-肩-右肩
            )
            # 左肘角度 (前臂与上臂的角度)
            angles['left_elbow'] = get_angle_3d(
                landmarks[11], landmarks[13], landmarks[15]  # 肩-肘-腕
            )
        
        # 右臂关节角度
        if all(landmarks[i].visibility > 0.5 for i in [12, 14, 16]):
            # 右肩角度
            angles['right_shoulder'] = get_angle_3d(
                landmarks[14], landmarks[12], landmarks[11]  # 肘-肩-左肩
            )
            # 右肘角度
            angles['right_elbow'] = get_angle_3d(
                landmarks[12], landmarks[14], landmarks[16]  # 肩-肘-腕
            )
            
        return angles

    def get_robot_coordinates(self, pose_landmarks, hand_landmarks_list=None):
        """获取机器人控制坐标（标准化到[-1,1]范围）"""
        robot_data = {
            'left_arm': {},
            'right_arm': {},
            'hands': []
        }
        
        if pose_landmarks:
            landmarks = pose_landmarks.landmark
            
            # 左臂坐标
            if landmarks[15].visibility > 0.5:  # 左腕可见
                robot_data['left_arm'] = {
                    'shoulder': [landmarks[11].x, landmarks[11].y, landmarks[11].z],
                    'elbow': [landmarks[13].x, landmarks[13].y, landmarks[13].z],
                    'wrist': [landmarks[15].x, landmarks[15].y, landmarks[15].z]
                }
            
            # 右臂坐标
            if landmarks[16].visibility > 0.5:  # 右腕可见
                robot_data['right_arm'] = {
                    'shoulder': [landmarks[12].x, landmarks[12].y, landmarks[12].z],
                    'elbow': [landmarks[14].x, landmarks[14].y, landmarks[14].z],
                    'wrist': [landmarks[16].x, landmarks[16].y, landmarks[16].z]
                }
        
        # 手部精确坐标
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                hand_data = {
                    'wrist': [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y],
                    'thumb_tip': [hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y],
                    'index_tip': [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y],
                    'middle_tip': [hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y],
                    'ring_tip': [hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y],
                    'pinky_tip': [hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y],
                }
                robot_data['hands'].append(hand_data)
        
        return robot_data

    def run(self):
        frame_count = 0
        start_time = time.time()
        show_hand_details = True
        
        while True:
            success, image = self.cap.read()
            if not success:
                print("警告：无法读取摄像头帧，跳过...")
                continue
                
            # 翻转图像
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # 姿态检测
            pose_results = self.pose.process(image_rgb)
            
            # 手部检测
            hand_results = None
            if show_hand_details:
                hand_results = self.hands.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # 处理检测结果
            arms_detected = False
            if pose_results.pose_landmarks:
                arms_detected = True
                landmarks = pose_results.pose_landmarks.landmark
                h, w, c = image.shape
                
                # 只绘制手臂关键点
                for name, idx in self.arm_keypoints.items():
                    landmark = landmarks[idx]
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        
                        # 根据关节类型设置不同颜色
                        if 'SHOULDER' in name:
                            color = (255, 0, 0)  # 蓝色
                        elif 'ELBOW' in name:
                            color = (0, 255, 0)  # 绿色
                        elif 'WRIST' in name:
                            color = (0, 0, 255)  # 红色
                        
                        cv2.circle(image, (x, y), 10, color, -1)
                        cv2.putText(image, name.split('_')[1][:3], (x-15, y-15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 绘制手臂连接线
                for connection in self.arm_connections:
                    start_idx, end_idx = connection
                    start_landmark = landmarks[start_idx]
                    end_landmark = landmarks[end_idx]
                    
                    if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                        start_x, start_y = int(start_landmark.x * w), int(start_landmark.y * h)
                        end_x, end_y = int(end_landmark.x * w), int(end_landmark.y * h)
                        cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 3)
                
                # 计算和显示关节角度
                angles = self.calculate_joint_angles(pose_results.pose_landmarks)
                y_offset = 90
                for joint, angle in angles.items():
                    cv2.putText(image, f"{joint}: {angle:.1f}°", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
            
            # 绘制手部细节
            hands_detected = 0
            if show_hand_details and hand_results and hand_results.multi_hand_landmarks:
                hands_detected = len(hand_results.multi_hand_landmarks)
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # 只绘制关键手指关键点
                    key_points = [0, 4, 8, 12, 16, 20]  # 腕、拇指尖、食指尖、中指尖、无名指尖、小指尖
                    h, w, c = image.shape
                    for idx in key_points:
                        landmark = hand_landmarks.landmark[idx]
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (x, y), 6, (0, 255, 255), -1)  # 黄色
            
            # 获取机器人控制数据
            robot_data = self.get_robot_coordinates(
                pose_results.pose_landmarks if pose_results.pose_landmarks else None,
                hand_results.multi_hand_landmarks if hand_results and hand_results.multi_hand_landmarks else None
            )
            
            # 显示状态信息
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(image, f"Arms: {'OK' if arms_detected else 'NO'} | Hands: {hands_detected}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(image, f"Hand Details: {'ON' if show_hand_details else 'OFF'} (Press 'h')", (10, 450), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 显示图像
            cv2.imshow('Robot Arm Control Detection', image)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"arm_control_{timestamp}.jpg"
                cv2.imwrite(filename, image)
                print(f"保存图像: {filename}")
            elif key == ord('h'):
                show_hand_details = not show_hand_details
                print(f"手部细节显示: {'开启' if show_hand_details else '关闭'}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("机器人手臂控制检测结束")

def main():
    try:
        detector = ArmControlDetection()
        detector.run()
    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
