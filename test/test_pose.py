#!/usr/bin/env python3
"""
MediaPipe pose detection test script
Full-body pose detection tailored for robot arm control
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys

class PoseDetectionTest:
    def __init__(self):
        print("Initializing MediaPipe pose detection...")
        
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=light, 1=full, 2=heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Define keypoints relevant to robot control
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
        
        # Initialize camera
        print("Connecting to camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Camera 0 failed; trying camera 1...")
            self.cap = cv2.VideoCapture(1)
            
        if not self.cap.isOpened():
            print("Error: unable to open any camera")
            sys.exit(1)
            
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ MediaPipe pose detection started")
        print("✓ Camera connected")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Stand in front of camera, raise both arms to test")
        print("  - Red dots: arm key joints")
        print("  - Green lines: body connections")

    def calculate_arm_angles(self, landmarks):
        """Compute arm joint angles (for robot control)"""
        def get_angle(p1, p2, p3):
            """Compute angle formed by three points"""
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        angles = {}
        
        # Left arm angles
        if all(landmarks.landmark[i].visibility > 0.5 for i in [11, 13, 15]):
            left_shoulder_angle = get_angle(
                landmarks.landmark[11],  # left shoulder
                landmarks.landmark[13],  # left elbow
                landmarks.landmark[15]   # left wrist
            )
            angles['left_elbow'] = left_shoulder_angle
        
        # Right arm angles
        if all(landmarks.landmark[i].visibility > 0.5 for i in [12, 14, 16]):
            right_shoulder_angle = get_angle(
                landmarks.landmark[12],  # right shoulder
                landmarks.landmark[14],  # right elbow
                landmarks.landmark[16]   # right wrist
            )
            angles['right_elbow'] = right_shoulder_angle
            
        return angles

    def run(self):
        frame_count = 0
        start_time = time.time()
        
        while True:
            success, image = self.cap.read()
            if not success:
                print("Warning: unable to read camera frame, skipping...")
                continue
                
            # Flip image
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Run pose detection
            results = self.pose.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw detection results
            pose_detected = False
            if results.pose_landmarks:
                pose_detected = True
                
                # Draw all pose connections
                self.mp_draw.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
                )
                
                # Highlight arm keypoints
                h, w, c = image.shape
                for name, idx in self.arm_landmarks.items():
                    landmark = results.pose_landmarks.landmark[idx]
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (x, y), 8, (0, 0, 255), -1)  # red dot
                        cv2.putText(image, name.split('_')[-1], (x-15, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                
                # Compute and display joint angles
                angles = self.calculate_arm_angles(results.pose_landmarks)
                y_offset = 90
                for joint, angle in angles.items():
                    cv2.putText(image, f"{joint}: {angle:.1f}°", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
            
            # Show status info
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
            
            # Show image
            cv2.imshow('MediaPipe Pose Detection for Robot Control', image)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pose_detection_{timestamp}.jpg"
                cv2.imwrite(filename, image)
                print(f"Saved image: {filename}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Pose detection test finished")

def main():
    try:
        detector = PoseDetectionTest()
        detector.run()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
