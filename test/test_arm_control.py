#!/usr/bin/env python3
"""
Custom robot arm control detection script
Detects and displays shoulder, elbow, wrist, and hand keypoints only
Designed specifically for dual UR3e robots
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys

class ArmControlDetection:
    def __init__(self):
        print("Initializing custom arm detection system...")
        
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe hands (more precise hand info)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        
        # Define keypoints needed for robot control (arms only)
        self.arm_keypoints = {
            # left arm
            'LEFT_SHOULDER': 11,
            'LEFT_ELBOW': 13,
            'LEFT_WRIST': 15,
            # right arm
            'RIGHT_SHOULDER': 12,
            'RIGHT_ELBOW': 14,
            'RIGHT_WRIST': 16,
        }
        
        # define connections for drawing
        self.arm_connections = [
            (11, 13),  # left shoulder to left elbow
            (13, 15),  # left elbow to left wrist
            (12, 14),  # right shoulder to right elbow  
            (14, 16),  # right elbow to right wrist
            (11, 12),  # left shoulder to right shoulder
        ]
        
        # Initialize camera
        print("Connecting to camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Camera 0 failed; trying camera 1...")
            self.cap = cv2.VideoCapture(1)
            
        if not self.cap.isOpened():
            print("Error: unable to open any camera")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ Custom arm detection started")
        print("✓ Camera connected")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'h' to toggle hand details")
        print("  - Blue: shoulder joints")
        print("  - Green: elbow joints") 
        print("  - Red: wrist joints")
        print("  - Yellow: hand keypoints")

    def calculate_joint_angles(self, pose_landmarks):
        """Compute joint angles needed for robot control"""
        def get_angle_3d(p1, p2, p3):
            """Compute angle among three points in 3D space"""
            v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        angles = {}
        landmarks = pose_landmarks.landmark
        
        # Left arm joint angles
        if all(landmarks[i].visibility > 0.5 for i in [11, 13, 15]):
            # Left shoulder angle (upper arm vs torso)
            angles['left_shoulder'] = get_angle_3d(
                landmarks[13], landmarks[11], landmarks[12]  
            )
            # Left elbow angle (forearm vs upper arm)
            angles['left_elbow'] = get_angle_3d(
                landmarks[11], landmarks[13], landmarks[15]  
            )
        
        # Right arm joint angles
        if all(landmarks[i].visibility > 0.5 for i in [12, 14, 16]):
            # Right shoulder angle
            angles['right_shoulder'] = get_angle_3d(
                landmarks[14], landmarks[12], landmarks[11]  
            )
            # Right elbow angle
            angles['right_elbow'] = get_angle_3d(
                landmarks[12], landmarks[14], landmarks[16]  
            )
            
        return angles

    def get_robot_coordinates(self, pose_landmarks, hand_landmarks_list=None):
        """Get robot control coordinates (normalized to [-1, 1])"""
        robot_data = {
            'left_arm': {},
            'right_arm': {},
            'hands': []
        }
        
        if pose_landmarks:
            landmarks = pose_landmarks.landmark
            
            # left arm coordinates
            if landmarks[15].visibility > 0.5:  # left wrist visible
                robot_data['left_arm'] = {
                    'shoulder': [landmarks[11].x, landmarks[11].y, landmarks[11].z],
                    'elbow': [landmarks[13].x, landmarks[13].y, landmarks[13].z],
                    'wrist': [landmarks[15].x, landmarks[15].y, landmarks[15].z]
                }
            
            # right arm coordinates
            if landmarks[16].visibility > 0.5:  # right wrist visible
                robot_data['right_arm'] = {
                    'shoulder': [landmarks[12].x, landmarks[12].y, landmarks[12].z],
                    'elbow': [landmarks[14].x, landmarks[14].y, landmarks[14].z],
                    'wrist': [landmarks[16].x, landmarks[16].y, landmarks[16].z]
                }
        
        # Precise hand coordinates
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
                print("Warning: Unable to read camera frame, skipping...")
                continue
                
            # Flip image
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Pose detection
            pose_results = self.pose.process(image_rgb)
            
            # Hand detection
            hand_results = None
            if show_hand_details:
                hand_results = self.hands.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Process detection results
            arms_detected = False
            if pose_results.pose_landmarks:
                arms_detected = True
                landmarks = pose_results.pose_landmarks.landmark
                h, w, c = image.shape
                
                # only draw arm keypoints
                for name, idx in self.arm_keypoints.items():
                    landmark = landmarks[idx]
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        
                        # Set different colors based on joint type
                        if 'SHOULDER' in name:
                            color = (255, 0, 0)  # blue
                        elif 'ELBOW' in name:
                            color = (0, 255, 0)  # green
                        elif 'WRIST' in name:
                            color = (0, 0, 255)  # red
                        
                        cv2.circle(image, (x, y), 10, color, -1)
                        cv2.putText(image, name.split('_')[1][:3], (x-15, y-15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw arm connection lines
                for connection in self.arm_connections:
                    start_idx, end_idx = connection
                    start_landmark = landmarks[start_idx]
                    end_landmark = landmarks[end_idx]
                    
                    if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                        start_x, start_y = int(start_landmark.x * w), int(start_landmark.y * h)
                        end_x, end_y = int(end_landmark.x * w), int(end_landmark.y * h)
                        cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 3)
                
                # Calculate and display joint angles
                angles = self.calculate_joint_angles(pose_results.pose_landmarks)
                y_offset = 90
                for joint, angle in angles.items():
                    cv2.putText(image, f"{joint}: {angle:.1f}°", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
            
            # Draw hand details
            hands_detected = 0
            if show_hand_details and hand_results and hand_results.multi_hand_landmarks:
                hands_detected = len(hand_results.multi_hand_landmarks)
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Only draw key finger landmarks
                    key_points = [0, 4, 8, 12, 16, 20]  # wrist, thumb tip, index tip, middle tip, ring tip, pinky tip
                    h, w, c = image.shape
                    for idx in key_points:
                        landmark = hand_landmarks.landmark[idx]
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (x, y), 6, (0, 255, 255), -1)  # yellow
            
            # Get robot control data
            robot_data = self.get_robot_coordinates(
                pose_results.pose_landmarks if pose_results.pose_landmarks else None,
                hand_results.multi_hand_landmarks if hand_results and hand_results.multi_hand_landmarks else None
            )
            
            # Show status information
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
            
            # Show image
            cv2.imshow('Robot Arm Control Detection', image)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"arm_control_{timestamp}.jpg"
                cv2.imwrite(filename, image)
                print(f"Saved image: {filename}")
            elif key == ord('h'):
                show_hand_details = not show_hand_details
                print(f"Hand details: {'ON' if show_hand_details else 'OFF'}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Robot arm control detection finished")

def main():
    try:
        detector = ArmControlDetection()
        detector.run()
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
