#!/usr/bin/env python3
"""
MediaPipe手部检测测试脚本
用于验证摄像头和MediaPipe是否正常工作
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class HandDetectionTest:
    def __init__(self):
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 最多检测2只手
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("错误：无法打开摄像头")
            exit()
            
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("MediaPipe手部检测测试启动")
        print("按 'q' 键退出")
        print("按 's' 键保存当前帧")

    def run(self):
        frame_count = 0
        start_time = time.time()
        
        while True:
            success, image = self.cap.read()
            if not success:
                print("忽略空摄像头帧")
                continue
                
            # 翻转图像（镜像效果）
            image = cv2.flip(image, 1)
            
            # 转换颜色格式（BGR -> RGB）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 提高性能
            image_rgb.flags.writeable = False
            
            # 进行手部检测
            results = self.hands.process(image_rgb)
            
            # 恢复图像可写状态
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # 绘制检测结果
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 绘制手部关键点和连线
                    self.mp_draw.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 获取手部信息
                    hand_info = results.multi_handedness[hand_idx]
                    hand_label = hand_info.classification[0].label  # 'Left' 或 'Right'
                    confidence = hand_info.classification[0].score
                    
                    # 获取手腕位置（关键点0）
                    wrist = hand_landmarks.landmark[0]
                    h, w, c = image.shape
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    
                    # 显示手部信息
                    cv2.putText(image, f"{hand_label} ({confidence:.2f})", 
                              (wrist_x - 50, wrist_y - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 显示关键点坐标（食指尖端 - 关键点8）
                    finger_tip = hand_landmarks.landmark[8]
                    tip_x, tip_y = int(finger_tip.x * w), int(finger_tip.y * h)
                    cv2.circle(image, (tip_x, tip_y), 10, (255, 0, 0), -1)
                    cv2.putText(image, f"({tip_x}, {tip_y})", 
                              (tip_x + 15, tip_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 计算FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 显示说明文字
            cv2.putText(image, "Press 'q' to quit, 's' to save", (10, 460), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示图像
            cv2.imshow('MediaPipe Hand Detection Test', image)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"hand_detection_{timestamp}.jpg"
                cv2.imwrite(filename, image)
                print(f"保存图像: {filename}")
        
        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("测试结束")

def main():
    try:
        detector = HandDetectionTest()
        detector.run()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
