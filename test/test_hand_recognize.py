#!/usr/bin/env python3
"""
Quick test script - verify the new teleoperation system
Test gesture recognition and 3D position control
"""

import cv2
import mediapipe as mp
import numpy as np

def test_hand_gesture_recognition():
    """Test gesture recognition"""
    print("=== Gesture Recognition Test ===")
    print("1. Open the camera")
    print("2. Open palm → should show 'OPEN'")
    print("3. Closed fist → should show 'CLOSED'")
    print("Press 'q' to quit\n")
    
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
        """Recognize gesture"""
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
            print("Warning: cannot read camera frame")
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
    print("✅ Gesture recognition test complete\n")


def test_arm_position_mapping():
    """Test arm position mapping"""
    print("=== Arm Position Mapping Test ===")
    print("Display wrist-to-workspace coordinate mapping")
    print("Press 'q' to quit\n")
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    # Define workspace boundaries
    workspace_x_min, workspace_x_max = 0.3, 0.9
    workspace_y_min, workspace_y_max = -0.4, 0.4
    workspace_z_min, workspace_z_max = 0.5, 1.2
    
    def calculate_robot_position(wrist, shoulder):
        """Compute robot position"""
        relative = wrist - shoulder
        
        x = np.interp(-relative[2], [-0.5, 0.3], [workspace_x_min, workspace_x_max])
            # Workspace parameters
        z = np.interp(1.0 - wrist[1], [0.3, 0.9], [workspace_z_min, workspace_z_max])
        
        x = np.clip(x, workspace_x_min, workspace_x_max)
        y = np.clip(y, workspace_y_min, workspace_y_max)
        z = np.clip(z, workspace_z_min, workspace_z_max)
        
        return x, y, z
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Warning: cannot read camera frame")
            break
        
        image = cv2.flip(image, 1)
        h, w, c = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            
            # left arm
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            left_wrist = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])
            
            if landmarks[15].visibility > 0.5:
                lx, ly, lz = calculate_robot_position(left_wrist, left_shoulder)
                cv2.putText(image, f"Left: X={lx:.2f} Y={ly:.2f} Z={lz:.2f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # right arm
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
                    # Left arm
            
            if landmarks[16].visibility > 0.5:
                rx, ry, rz = calculate_robot_position(right_wrist, right_shoulder)
                cv2.putText(image, f"Right: X={rx:.2f} Y={ry:.2f} Z={rz:.2f}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(image, "Press 'q' to quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
                    # Right arm
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Position mapping test complete\n")


def main():
    print("\n" + "="*50)
    print("  Teleoperation System Functional Tests")
    print("="*50 + "\n")
    
    print("Select a test:")
    print("1. Gesture recognition test")
    print("2. Position mapping test")
    print("3. Run all tests")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        test_hand_gesture_recognition()
    elif choice == '2':
        test_arm_position_mapping()
    elif choice == '3':
        test_hand_gesture_recognition()
        test_arm_position_mapping()
    else:
        print("Invalid choice")
    
    print("\n" + "="*50)
    print("  Tests complete!")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
