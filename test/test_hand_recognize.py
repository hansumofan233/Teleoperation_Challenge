#!/usr/bin/env python3
"""
快速测试脚本 - 验证新的遥操作系统
测试手势识别和3D位置控制
"""

import cv2
import mediapipe as mp
import numpy as np

def test_hand_gesture_recognition():
    """测试手势识别功能"""
    print("=== 手势识别测试 ===")
    print("1. 打开摄像头")
    print("2. 张开手掌 → 应显示 'OPEN'")
    print("3. 握拳 → 应显示 'CLOSED'")
    print("按 'q' 退出测试\n")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    def recognize_gesture(hand_landmarks):
        """识别手势"""
        landmarks = hand_landmarks.landmark
        palm_center = landmarks[0]
        
        finger_tips = [8, 12, 16, 20]
        finger_mcp = [5, 9, 13, 17]
        
        extended_fingers = 0
        for tip_idx, mcp_idx in zip(finger_tips, finger_mcp):
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            
            tip_dist = np.sqrt((tip.x - palm_center.x)**2 + (tip.y - palm_center.y)**2)
            mcp_dist = np.sqrt((mcp.x - palm_center.x)**2 + (mcp.y - palm_center.y)**2)
            
            if tip_dist > mcp_dist * 1.3:
                extended_fingers += 1
        
        if extended_fingers >= 3:
            return 'OPEN'
        elif extended_fingers <= 1:
            return 'CLOSED'
        else:
            return 'UNCERTAIN'
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("警告: 无法读取摄像头画面")
            break
        
        image = cv2.flip(image, 1)
        h, w, c = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                gesture = recognize_gesture(hand_landmarks)
                hand_label = handedness.classification[0].label
                
                text = f"{hand_label}: {gesture}"
                color = (0, 255, 0) if gesture == 'OPEN' else (0, 0, 255) if gesture == 'CLOSED' else (255, 255, 0)
                
                cv2.putText(image, text, (10, 50 if hand_label == 'Left' else 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.putText(image, "Press 'q' to quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Gesture Recognition Test', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 手势识别测试完成\n")


def test_arm_position_mapping():
    """测试手臂位置映射"""
    print("=== 手臂位置映射测试 ===")
    print("显示手腕位置映射到机械臂工作空间的坐标")
    print("按 'q' 退出测试\n")
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    # 工作空间参数
    workspace_x_min, workspace_x_max = 0.3, 0.9
    workspace_y_min, workspace_y_max = -0.4, 0.4
    workspace_z_min, workspace_z_max = 0.5, 1.2
    
    def calculate_robot_position(wrist, shoulder):
        """计算机器人位置"""
        relative = wrist - shoulder
        
        x = np.interp(-relative[2], [-0.5, 0.3], [workspace_x_min, workspace_x_max])
        y = np.interp(relative[0], [-0.2, 0.5], [workspace_y_min, workspace_y_max])
        z = np.interp(1.0 - wrist[1], [0.3, 0.9], [workspace_z_min, workspace_z_max])
        
        x = np.clip(x, workspace_x_min, workspace_x_max)
        y = np.clip(y, workspace_y_min, workspace_y_max)
        z = np.clip(z, workspace_z_min, workspace_z_max)
        
        return x, y, z
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("警告: 无法读取摄像头画面")
            break
        
        image = cv2.flip(image, 1)
        h, w, c = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            
            # 左臂
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            left_wrist = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])
            
            if landmarks[15].visibility > 0.5:
                lx, ly, lz = calculate_robot_position(left_wrist, left_shoulder)
                cv2.putText(image, f"Left: X={lx:.2f} Y={ly:.2f} Z={lz:.2f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 右臂
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            right_wrist = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
            
            if landmarks[16].visibility > 0.5:
                rx, ry, rz = calculate_robot_position(right_wrist, right_shoulder)
                cv2.putText(image, f"Right: X={rx:.2f} Y={ry:.2f} Z={rz:.2f}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(image, "Press 'q' to quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Position Mapping Test', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 位置映射测试完成\n")


def main():
    print("\n" + "="*50)
    print("  遥操作系统功能测试")
    print("="*50 + "\n")
    
    print("选择测试:")
    print("1. 手势识别测试")
    print("2. 位置映射测试")
    print("3. 全部测试")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        test_hand_gesture_recognition()
    elif choice == '2':
        test_arm_position_mapping()
    elif choice == '3':
        test_hand_gesture_recognition()
        test_arm_position_mapping()
    else:
        print("无效选择")
    
    print("\n" + "="*50)
    print("  测试完成!")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
