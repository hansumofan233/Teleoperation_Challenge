#!/usr/bin/env python3
"""
MediaPipe姿态检测测试脚本
专门用于机器人手臂控制的全身姿态检测
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys

class PoseDetectionTest:
    def __init__(self):
        print("正在初始化MediaPipe姿态检测...")
        
        # 初始化MediaPipe姿态检测
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=轻量, 1=完整, 2=重型
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 定义机器人控制相关的关键点
        self.arm_landmarks = {
            'LEFT_SHOULDER': 11,
            'LEFT_ELBOW': 13,
            'LEFT_WRIST': 15,
            'LEFT_PINKY': 17,
            'LEFT_INDEX': 19,
            'LEFT_THUMB': 21,
            'RIGHT_SHOULDER': 12,
            'RIGHT_ELBOW': 14,
            'RIGHT_WRIST': 16,
            'RIGHT_PINKY': 18,
            'RIGHT_INDEX': 20,
            'RIGHT_THUMB': 22,
        }
        
        # 初始化摄像头
        print("正在连接摄像头...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("摄像头0无法打开，尝试摄像头1...")
            self.cap = cv2.VideoCapture(1)
            
        if not self.cap.isOpened():
            print("错误：无法打开任何摄像头")
            sys.exit(1)
            
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ MediaPipe姿态检测启动成功")
        print("✓ 摄像头连接成功")
        print("控制说明：")
        print("  - 按 'q' 键退出")
        print("  - 按 's' 键保存当前帧")
        print("  - 站到摄像头前，张开双臂测试检测效果")
        print("  - 红色点：手臂关键关节")
        print("  - 绿色线：身体连接")

    def calculate_arm_angles(self, landmarks):
        """计算手臂关节角度（用于机器人控制）"""
        def get_angle(p1, p2, p3):
            """计算三点构成的角度"""
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        angles = {}
        
        # 左臂角度
        if all(landmarks.landmark[i].visibility > 0.5 for i in [11, 13, 15]):
            left_shoulder_angle = get_angle(
                landmarks.landmark[11],  # 左肩
                landmarks.landmark[13],  # 左肘
                landmarks.landmark[15]   # 左腕
            )
            angles['left_elbow'] = left_shoulder_angle
        
        # 右臂角度
        if all(landmarks.landmark[i].visibility > 0.5 for i in [12, 14, 16]):
            right_shoulder_angle = get_angle(
                landmarks.landmark[12],  # 右肩
                landmarks.landmark[14],  # 右肘
                landmarks.landmark[16]   # 右腕
            )
            angles['right_elbow'] = right_shoulder_angle
            
        return angles

    def run(self):
        frame_count = 0
        start_time = time.time()
        
        while True:
            success, image = self.cap.read()
            if not success:
                print("警告：无法读取摄像头帧，跳过...")
                continue
                
            # 翻转图像
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # 进行姿态检测
            results = self.pose.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # 绘制检测结果
            pose_detected = False
            if results.pose_landmarks:
                pose_detected = True
                
                # 绘制所有姿态连接
                self.mp_draw.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
                )
                
                # 突出显示手臂关键点
                h, w, c = image.shape
                for name, idx in self.arm_landmarks.items():
                    landmark = results.pose_landmarks.landmark[idx]
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (x, y), 8, (0, 0, 255), -1)  # 红色圆点
                        cv2.putText(image, name.split('_')[-1], (x-15, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                
                # 计算并显示关节角度
                angles = self.calculate_arm_angles(results.pose_landmarks)
                y_offset = 90
                for joint, angle in angles.items():
                    cv2.putText(image, f"{joint}: {angle:.1f}°", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
            
            # 显示状态信息
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(image, f"Pose Detected: {'Yes' if pose_detected else 'No'}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(image, "Stand back, raise both arms", (10, 450), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示图像
            cv2.imshow('MediaPipe Pose Detection for Robot Control', image)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pose_detection_{timestamp}.jpg"
                cv2.imwrite(filename, image)
                print(f"保存图像: {filename}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("姿态检测测试结束")

def main():
    try:
        detector = PoseDetectionTest()
        detector.run()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
