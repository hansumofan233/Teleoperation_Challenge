#!/usr/bin/env python3
"""
Camera-based teleoperation for end-effector control
Implements 2D planar motion + 1D yaw rotation
Uses MediaPipe to capture arm motion and map to robot end-effector
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float64MultiArray, Header
from std_msgs.msg import Empty
import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class TeleopEndEffectorControl(Node):
    def __init__(self):
        super().__init__('teleop_endeffector_control')
        
        # ROS2 Publishers - publish end-effector target positions
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
        
        # Parameter setup
        self.declare_parameter('workspace_x_min', 0.3)  # Workspace X range (m) - away from base
        self.declare_parameter('workspace_x_max', 0.9)  # Extend forward range
        self.declare_parameter('workspace_y_min', -0.4)  # Workspace Y range (m) - widen left/right
        self.declare_parameter('workspace_y_max', 0.4)
        self.declare_parameter('workspace_z', 0.9)  # Fixed Z height (m)
        self.declare_parameter('rotation_sensitivity', 2.0)  # Rotation sensitivity
        self.declare_parameter('position_smoothing', 0.15)  # Position smoothing (lower to reduce jitter)
        self.declare_parameter('camera_id', 0)
        # With X/Z-only control, lateral Y is fixed at 0 (no lateral offset)
        self.declare_parameter('fixed_y_left', 0.0)
        self.declare_parameter('fixed_y_right', 0.0)
        # Relative displacement scaling (map normalized image coords to world range)
        self.declare_parameter('relative_frames_for_anchor', 5)
        # Pixel→meter scaling modes:
        # - 'workspace': use workspace range scaling (normalized 0-1 × range)
        # - 'fixed': use fixed coefficients (meters per pixel)
        self.declare_parameter('scale_mode', 'workspace')
        self.declare_parameter('scale_x_m_per_px', 0.0015)  # Meters per pixel for X (used when scale_mode='fixed')
        self.declare_parameter('scale_z_m_per_px', 0.0015)  # Meters per pixel for Z (used when scale_mode='fixed')
        self.declare_parameter('yaw_smoothing', 0.2)
        self.declare_parameter('yaw_deadband_deg', 2.0)
        # Debug: sequence-based periodic logging
        self.declare_parameter('debug_seq_enable', True)
        self.declare_parameter('debug_log_every_n', 30)
        # Desensitization thresholds
        self.declare_parameter('min_cam_delta', 0.01)  # min image-space movement (lower → more sensitive)
        self.declare_parameter('dominant_ratio', 1.5)  # dominant-arm movement ratio threshold
        self.declare_parameter('min_world_delta_x', 0.005)  # min world ΔX (m)
        self.declare_parameter('min_world_delta_z', 0.003)  # min world ΔZ (m), improve small up/down sensitivity
        # Display scaling (visual only, does not affect computation)
        # Set to sqrt(2) ≈ 1.414 to get ~2x area
        self.declare_parameter('display_scale', 1.41421356)
        
        # Get parameters
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
        self.display_scale = float(self.get_parameter('display_scale').value)
        
        # Gripper command publishers
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
        
        # Init MediaPipe Pose and Hands
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
        
        # Pose keypoint indices
        self.keypoints = {
            'LEFT_SHOULDER': 11,
            'LEFT_ELBOW': 13,
            'LEFT_WRIST': 15,
            'RIGHT_SHOULDER': 12,
            'RIGHT_ELBOW': 14,
            'RIGHT_WRIST': 16,
        }
        
        # Z-axis control range (based on wrist height)
        self.declare_parameter('workspace_z_min', 0.5)  # Minimum height
        self.declare_parameter('workspace_z_max', 1.2)  # Maximum height
        self.workspace_z_min = self.get_parameter('workspace_z_min').value
        self.workspace_z_max = self.get_parameter('workspace_z_max').value
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {camera_id}')
            raise RuntimeError('Camera initialization failed')
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initial positions (used for reset)
        self.INITIAL_LEFT_POSITION = np.array([0.6, 0.3, 0.7])
        self.INITIAL_RIGHT_POSITION = np.array([0.6, -0.3, 0.7])
        
        # Arm state variables
        self.left_arm_state = {
            'position': self.INITIAL_LEFT_POSITION.copy(),  # Left arm initial absolute position
            'prev_position': None,
            'yaw': 0.0,
            'prev_yaw': None,
            'gripper_open': True,  # gripper state
            'prev_gripper_open': True,
            # Relative displacement anchor and current relative values (image wrist point)
            'anchor_cam_xy': None,
            'anchor_counter': 0,
            'relative_world_dx': 0.0,
            'relative_world_dz': 0.0,
            # Most recent relative target used for logging (dx, dz, yaw)
            'last_relative': (0.0, 0.0, 0.0)
        }
        self.right_arm_state = {
            'position': self.INITIAL_RIGHT_POSITION.copy(),  # Right arm initial absolute position
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
        
        # Gesture recognition buffer (debounce)
        self.gesture_buffer = {
            'left': [],  # Store recent N-frame gestures
            'right': []
        }
        self.gesture_buffer_size = 5  # Buffer size
        
        # Arm visibility debounce - must be invisible for N consecutive frames to stop sending commands
        self.visibility_buffer = {
            'left': [],   # Store visibility states
            'right': []
        }
        self.visibility_buffer_size = 5  # Require 5 consecutive frames
        self.visibility_threshold = 0.8  # ≥80% frames visible → stably visible
        
        # Create timer to process video stream
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30Hz
        
        self.get_logger().info('Teleop end-effector control node started')
        self.get_logger().info(f'Workspace: X[{self.workspace_x_min}, {self.workspace_x_max}], '
                      f'Y[{self.workspace_y_min}, {self.workspace_y_max}], '
                      f'Z[{self.workspace_z_min}, {self.workspace_z_max}]')
        self.get_logger().info('=== Usage ===')
        self.get_logger().info('Arm control: 3DOF (X/Z/yaw) + gestures')
        self.get_logger().info('Mapping: wrist X → EE X, wrist height Y → EE Z, wrist rotation → EE yaw')
        self.get_logger().info('  - Move arm left/right → robot moves left/right in plane')
        self.get_logger().info('  - Raise arm → robot rises')
        self.get_logger().info('  - Rotate palm facing screen → robot rotates wrist')
        self.get_logger().info('Gesture control:')
        self.get_logger().info('  - Open palm → open gripper')
        self.get_logger().info('  - Closed fist → close gripper')
        # Subscribe to unified reset topic (triggered by RViz button/service)
        self.reset_sub = self.create_subscription(Empty, '/teleop/reset', self.on_reset_msg, 10)
        self.get_logger().info('Reset: use the RViz Reset button or publish /teleop/reset')
        if self.debug_seq_enable:
            self.get_logger().info(f'Debug seq enabled: print every {self.debug_log_every_n} frames (based on seq)')
        else:
            self.get_logger().info('Debug seq disabled: print based on internal counter')
    
    def _get_pose_wrist_pixels(self, pose_landmarks, image_shape):
        """Return visible wrist pixel coords: {'left': (x,y) or None, 'right': (x,y) or None}"""
        if pose_landmarks is None:
            return {'left': None, 'right': None}
        h, w, _ = image_shape
        # Note: to match mirroring strategy in calculate_endeffector_pose,
        # the "left arm" reference wrist pixel uses RIGHT_WRIST, and vice versa.
        l_idx = self.keypoints['RIGHT_WRIST']  # Mapped to left arm
        r_idx = self.keypoints['LEFT_WRIST']   # Mapped to right arm
        lm = pose_landmarks.landmark
        left = None
        right = None
        if lm[l_idx].visibility > 0.5:
            left = (float(lm[l_idx].x * w), float(lm[l_idx].y * h))
        if lm[r_idx].visibility > 0.5:
            right = (float(lm[r_idx].x * w), float(lm[r_idx].y * h))
        return {'left': left, 'right': right}

    def _assign_arm_for_hand(self, hand_landmarks, wrists_px, image_shape, fallback_label):
        """Assign left/right arm based on nearest wrist; fallback to handedness when both wrists missing."""
        if hand_landmarks is None:
            return 'left' if fallback_label == 'Left' else 'right'
        h, w, _ = image_shape
        hx = float(hand_landmarks.landmark[0].x * w)
        hy = float(hand_landmarks.landmark[0].y * h)
        left_px = wrists_px.get('left')
        right_px = wrists_px.get('right')
        # Both wrists invisible → fallback to MediaPipe handedness
        if left_px is None and right_px is None:
            return 'left' if fallback_label == 'Left' else 'right'
        # Only one side visible
        if right_px is None:
            return 'left'
        if left_px is None:
            return 'right'
        # Compare distances, choose nearest wrist
        dl = (hx - left_px[0])**2 + (hy - left_px[1])**2
        dr = (hx - right_px[0])**2 + (hy - right_px[1])**2
        return 'left' if dl <= dr else 'right'

    def reset_to_initial_position(self):
        """Reset both arms to initial positions"""
        self.get_logger().info('Resetting arms to initial positions...')
        
        # Reset left arm
        self.left_arm_state['position'] = self.INITIAL_LEFT_POSITION.copy()
        self.left_arm_state['prev_position'] = None
        self.left_arm_state['gripper_open'] = True
        # Clear anchors/relative
        self.left_arm_state['anchor_cam_xy'] = None
        self.left_arm_state['anchor_counter'] = 0
        self.left_arm_state['relative_world_dx'] = 0.0
        self.left_arm_state['relative_world_dz'] = 0.0
        self.left_arm_state['last_relative'] = (0.0, 0.0, 0.0)
        self.left_arm_state['yaw'] = 0.0
        self.left_arm_state['prev_yaw'] = None

        self.publish_endeffector_command('left')
        self.publish_gripper_command('left', True)
        
        # Reset right arm
        self.right_arm_state['position'] = self.INITIAL_RIGHT_POSITION.copy()
        self.right_arm_state['prev_position'] = None
        self.right_arm_state['gripper_open'] = True
        # Clear anchors/relative
        self.right_arm_state['anchor_cam_xy'] = None
        self.right_arm_state['anchor_counter'] = 0
        self.right_arm_state['relative_world_dx'] = 0.0
        self.right_arm_state['relative_world_dz'] = 0.0
        self.right_arm_state['last_relative'] = (0.0, 0.0, 0.0)
        self.right_arm_state['yaw'] = 0.0
        self.right_arm_state['prev_yaw'] = None

        self.publish_endeffector_command('right')
        self.publish_gripper_command('right', True)
        
        # Clear gesture buffers
        self.gesture_buffer['left'].clear()
        self.gesture_buffer['right'].clear()
        
        self.get_logger().info('✅ Reset to initial positions: Left(0.6, 0.3, 0.7), Right(0.6, -0.3, 0.7)')

    def on_reset_msg(self, _msg: Empty):
        """Receive unified reset signal: clear anchors/relative and return to initial pose"""
        self.get_logger().info('Received /teleop/reset, resetting visual reference and target baseline')
        self.reset_to_initial_position()

    def estimate_wrist_yaw(self, hand_landmarks):
        """Estimate wrist yaw (radians) from hand landmarks; coarse accuracy with noise reduction"""
        if hand_landmarks is None:
            return None
        lm = hand_landmarks.landmark
        # Use average of vectors from wrist (0) to index MCP (5) and middle MCP (9) as palm orientation
        try:
            wx, wy = lm[0].x, lm[0].y
            v1 = np.array([lm[9].x - wx, lm[9].y - wy])
            v2 = np.array([lm[5].x - wx, lm[5].y - wy])
            v = (v1 + v2) * 0.5
            if np.linalg.norm(v) < 1e-6:
                return None
            # Image coordinates: x→right, y→down; atan2(dy, dx)
            raw_angle = np.arctan2(v[1], v[0])
            # Map angle to [-pi, pi]
            if raw_angle > np.pi:
                raw_angle -= 2*np.pi
            if raw_angle < -np.pi:
                raw_angle += 2*np.pi
            return float(raw_angle)
        except Exception:
            return None

    def smooth_yaw(self, new_yaw, prev_yaw):
        """Apply first-order low-pass filter and small deadband to yaw"""
        if new_yaw is None:
            return prev_yaw if prev_yaw is not None else 0.0
        if prev_yaw is None:
            prev_yaw = new_yaw
        # Small deadband: ignore tiny changes
        if abs(new_yaw - prev_yaw) < self.yaw_deadband:
            return prev_yaw
        return self.yaw_alpha * new_yaw + (1.0 - self.yaw_alpha) * prev_yaw

    def update_relative_from_cam(self, arm_state, cam_xy):
        """
        Compute relative displacement from image wrist point:
        - Anchor point a: first detection of wrist; record and count; stable when counter reaches anchor_wait_frames
        - Relative b-a: current wrist minus anchor, mapped to world (dx, dz)
        - Scaling:
            * scale_mode='workspace': normalized delta × (X/Z workspace range)
            * scale_mode='fixed': pixel delta × (scale_x_m_per_px / scale_z_m_per_px)
        - Z: image y smaller → higher; so dz = -(cy - ay) × scaling
        - Desensitization: min_cam_delta, min_world_delta_x/z thresholds
        - Record last_relative=(dx, dz, yaw) for logging
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
        # Anchor stabilization counter
        if arm_state['anchor_counter'] < self.anchor_wait_frames:
            arm_state['anchor_counter'] += 1
        # Relative to first visible anchor a: image delta → world delta
        ax, ay = arm_state['anchor_cam_xy']
        cx, cy = cam_xy
        # Desensitize: if camera delta too small, keep previous relative displacement
        cam_dx = cx - ax
        cam_dy = cy - ay
        cam_dist = np.hypot(cam_dx, cam_dy)
        if cam_dist < self.min_cam_delta:
            dx_world = arm_state['relative_world_dx']
            dz_world = arm_state['relative_world_dz']
        else:
            if self.scale_mode == 'fixed':
                # Use raw pixel delta × fixed scaling coefficients
                dx_world = cam_dx * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale_x_m_per_px
                dz_world = -cam_dy * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale_z_m_per_px
            else:
                # Use normalized delta × workspace range
                dx_world = cam_dx * (self.workspace_x_max - self.workspace_x_min)
                dz_world = -cam_dy * (self.workspace_z_max - self.workspace_z_min)
            # World-space deadband
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
        """Recognize gesture: return 'open' or 'closed'"""
        if hand_landmarks is None:
            return None
        
        landmarks = hand_landmarks.landmark
        
        # Estimate finger extension
        # Use distance between fingertips and palm center
        palm_center = landmarks[0]  # Wrist
        
        # Check four fingers (index/middle/ring/pinky) straightness
        finger_tips = [8, 12, 16, 20]  # Fingertips
        finger_mcp = [5, 9, 13, 17]   # Finger MCP joints
        
        extended_fingers = 0
        for tip_idx, mcp_idx in zip(finger_tips, finger_mcp):
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            palm = palm_center
            
            # Compute distances
            tip_dist = np.sqrt((tip.x - palm.x)**2 + (tip.y - palm.y)**2)
            mcp_dist = np.sqrt((mcp.x - palm.x)**2 + (mcp.y - palm.y)**2)
            
            # If tip distance » MCP distance → finger extended
            if tip_dist > mcp_dist * 1.3:
                extended_fingers += 1
        
        # Determine gesture
        if extended_fingers >= 3:  # ≥3 fingers extended → open palm
            return 'open'
        elif extended_fingers <= 1:  # ≤1 finger extended → closed fist
            return 'closed'
        else:
            return None  # Intermediate/uncertain
    
    def update_gesture_buffer(self, arm, gesture):
        """Update gesture buffer and return a stable gesture (debounced)"""
        buffer = self.gesture_buffer[arm]
        
        if gesture is not None:
            buffer.append(gesture)
        
        # Maintain buffer size
        if len(buffer) > self.gesture_buffer_size:
            buffer.pop(0)
        
        # If majority in buffer is the same gesture, return it
        if len(buffer) >= self.gesture_buffer_size:
            open_count = buffer.count('open')
            closed_count = buffer.count('closed')
            
            if open_count > self.gesture_buffer_size * 0.7:
                return True  # Open
            elif closed_count > self.gesture_buffer_size * 0.7:
                return False  # Closed
        
        return None  # Uncertain
    
    def calculate_endeffector_pose(self, pose_landmarks, arm='left'):
        """
        Compute end-effector pose (3D position: X, Y, Z) and image wrist point

        Mapping (as required):
        - X-axis: wrist image X → robot EE X (forward/back)
        - Z-axis: wrist image Y height → robot EE Z (up/down)
        - Y-axis: fixed offset for the selected arm side
        """
        landmarks = pose_landmarks.landmark
        
        # Select keypoints for the corresponding arm (mirrored: left arm uses right-side keypoints)
        if arm == 'left':
            shoulder_idx = self.keypoints['RIGHT_SHOULDER']
            wrist_idx = self.keypoints['RIGHT_WRIST']
        else:
            shoulder_idx = self.keypoints['LEFT_SHOULDER']
            wrist_idx = self.keypoints['LEFT_WRIST']
        
        # Check visibility (higher thresholds → stricter)
        if landmarks[wrist_idx].visibility < 0.7 or landmarks[shoulder_idx].visibility < 0.7:
            return None
        
        # Get image coordinates (0-1)
        wrist = np.array([landmarks[wrist_idx].x, landmarks[wrist_idx].y])
        # Sanity check for reasonable range
        if not (0.05 < wrist[0] < 0.95 and 0.05 < wrist[1] < 0.95):
            return None

        # X: map image x from [0,1] → workspace [x_min, x_max]
        x = np.interp(wrist[0], [0.0, 1.0], [self.workspace_x_min, self.workspace_x_max])
        # Z: smaller image y → higher; map to [z_min, z_max]
        z = np.interp(1.0 - wrist[1], [0.0, 1.0], [self.workspace_z_min, self.workspace_z_max])
        # Fixed Y
        y = self.fixed_y_left if arm == 'left' else self.fixed_y_right
        
        # Clamp within workspace limits
        x = np.clip(x, self.workspace_x_min, self.workspace_x_max)
        y = np.clip(y, self.workspace_y_min, self.workspace_y_max)
        z = np.clip(z, self.workspace_z_min, self.workspace_z_max)
        
        position = np.array([x, y, z])
        # Return image wrist point for relative displacement
        return {'position': position, 'cam_wrist': (float(wrist[0]), float(wrist[1]))}
    
    def smooth_motion(self, current, previous, alpha=None):
        """Motion smoothing filter"""
        if alpha is None:
            alpha = self.smoothing
        if previous is None:
            return current
        return alpha * current + (1 - alpha) * previous
    
    def publish_endeffector_command(self, arm='left'):
        """Publish relative EE target: [ΔX, ΔZ, ΔYaw, (optional) seq] and log"""
        # Standard publishing: left arm → left topic, right arm → right topic
        if arm == 'left':
            state = self.left_arm_state
            publisher = self.left_ee_pub
        else:
            state = self.right_arm_state
            publisher = self.right_ee_pub
        
        # Build message: [ΔX, ΔZ, ΔYaw, (optional) seq]
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
        
        # Debug info: periodically log published target
        if self.debug_seq_enable:
            seq = int(payload[3]) if len(payload) > 3 else -1
            if seq > 0 and seq % self.debug_log_every_n == 0:
                self.get_logger().info(f'[{arm} arm] Relative EE target: dx={rel_dx:.3f}m, dz={rel_dz:.3f}m, yaw={rel_yaw:.2f}rad (seq={seq})')
        else:
            if not hasattr(self, f'_{arm}_pub_count'):
                setattr(self, f'_{arm}_pub_count', 0)
            count = getattr(self, f'_{arm}_pub_count') + 1
            setattr(self, f'_{arm}_pub_count', count)
            if count % self.debug_log_every_n == 0:
                self.get_logger().info(f'[{arm} arm] Relative EE target: dx={rel_dx:.3f}m, dz={rel_dz:.3f}m, yaw={rel_yaw:.2f}rad')
    
    def publish_gripper_command(self, arm='left', open_gripper=True):
        """Publish gripper command"""
        # Standard publishing: left arm → left gripper topic, right arm → right gripper topic
        if arm == 'left':
            publisher = self.left_gripper_pub
            state = self.left_arm_state
        else:
            publisher = self.right_gripper_pub
            state = self.right_arm_state
        
        # Avoid resending identical command
        if state['prev_gripper_open'] == open_gripper:
            return
        
        # Build message: [gripper_position] (0.0=closed, 1.0=open)
        msg = Float64MultiArray()
        msg.data = [1.0 if open_gripper else 0.0]
        
        publisher.publish(msg)
        state['prev_gripper_open'] = open_gripper
        
        self.get_logger().info(f'{arm} gripper: {"Open" if open_gripper else "Closed"}')
    
    def draw_visualization(self, image, pose_landmarks, hands_results):
        """Draw visualization overlays"""
        h, w, c = image.shape
        
        # Draw pose skeleton
        if pose_landmarks is not None:
            landmarks = pose_landmarks.landmark
            
            # Draw arm bones
            arm_connections = [
                (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),  # Right arm
                (11, 12)  # Shoulder connection
            ]
            
            for connection in arm_connections:
                start_idx, end_idx = connection
                if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
                    start_x = int(landmarks[start_idx].x * w)
                    start_y = int(landmarks[start_idx].y * h)
                    end_x = int(landmarks[end_idx].x * w)
                    end_y = int(landmarks[end_idx].y * h)
                    cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 3)
            
            # Draw keypoints
            for name, idx in self.keypoints.items():
                if landmarks[idx].visibility > 0.5:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    
                    if 'SHOULDER' in name:
                        color = (255, 0, 0)  # Blue
                    elif 'ELBOW' in name:
                        color = (0, 255, 0)  # Green
                    else:  # WRIST
                        color = (0, 0, 255)  # Red
                    
                    cv2.circle(image, (x, y), 8, color, -1)
        
        # Draw hand keypoints
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        
        # Display status information
        info_y = 30
        left_gripper_status = "Open" if self.left_arm_state['gripper_open'] else "Closed"
        cv2.putText(image, f"Left: ({self.left_arm_state['position'][0]:.2f}, "
                   f"{self.left_arm_state['position'][1]:.2f}, {self.left_arm_state['position'][2]:.2f}) "
                   f"Gripper: {left_gripper_status}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        info_y += 25
        
        right_gripper_status = "Open" if self.right_arm_state['gripper_open'] else "Closed"
        cv2.putText(image, f"Right: ({self.right_arm_state['position'][0]:.2f}, "
                   f"{self.right_arm_state['position'][1]:.2f}, {self.right_arm_state['position'][2]:.2f}) "
                   f"Gripper: {right_gripper_status}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        info_y += 25
        
        # Control instructions
        cv2.putText(image, "q:Quit | s:Save | Reset via RViz button | Open Palm=Open | Fist=Close", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def process_frame(self):
        """Main loop for processing video frames"""
        success, image = self.cap.read()
        if not success:
            return
        
        # Flip image
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Pose and hand detection
        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Process pose detection results
        if pose_results.pose_landmarks:
            # Compute end-effector poses
            left_pose = self.calculate_endeffector_pose(pose_results.pose_landmarks, 'left')
            right_pose = self.calculate_endeffector_pose(pose_results.pose_landmarks, 'right')
            
            # Update visibility buffer (debounced)
            self.visibility_buffer['left'].append(left_pose is not None)
            self.visibility_buffer['right'].append(right_pose is not None)
            
            # Maintain buffer size
            if len(self.visibility_buffer['left']) > self.visibility_buffer_size:
                self.visibility_buffer['left'].pop(0)
            if len(self.visibility_buffer['right']) > self.visibility_buffer_size:
                self.visibility_buffer['right'].pop(0)
            
            # Determine stable visibility: must reach threshold ratio to be considered visible
            left_is_visible = sum(self.visibility_buffer['left']) >= len(self.visibility_buffer['left']) * self.visibility_threshold
            right_is_visible = sum(self.visibility_buffer['right']) >= len(self.visibility_buffer['right']) * self.visibility_threshold
            
            # Update left arm position (with desensitization)
            if left_pose is not None and left_is_visible:
                self.left_arm_state['position'] = self.smooth_motion(
                    left_pose['position'], 
                    self.left_arm_state['prev_position']
                )
                self.left_arm_state['prev_position'] = self.left_arm_state['position'].copy()
                # Update relative displacement (based on image wrist point)
                self.update_relative_from_cam(self.left_arm_state, left_pose.get('cam_wrist'))
                
                # Publish control command
                self.publish_endeffector_command('left')
                
                # Reset warning flag and notify recovery
                if hasattr(self, '_left_invisible_warned'):
                    del self._left_invisible_warned
                    self.get_logger().info('✅ Left arm visible again, resuming tracking')
            else:
                # When arm not visible, hold current position; do not update or publish
                if not hasattr(self, '_left_invisible_warned'):
                    self._left_invisible_warned = True
                    self.get_logger().info('⚠️  Left arm not visible; holding position')
            
            # Update right arm position (with desensitization)
            if right_pose is not None and right_is_visible:
                self.right_arm_state['position'] = self.smooth_motion(
                    right_pose['position'],
                    self.right_arm_state['prev_position']
                )
                self.right_arm_state['prev_position'] = self.right_arm_state['position'].copy()
                self.update_relative_from_cam(self.right_arm_state, right_pose.get('cam_wrist'))
                
                # Publish control command
                self.publish_endeffector_command('right')
                
                # Reset warning flag and notify recovery
                if hasattr(self, '_right_invisible_warned'):
                    del self._right_invisible_warned
                    self.get_logger().info('✅ Right arm visible again, resuming tracking')
            else:
                # When arm not visible, hold current position; do not update or publish
                if not hasattr(self, '_right_invisible_warned'):
                    self._right_invisible_warned = True
                    self.get_logger().info('⚠️  Right arm not visible; holding position')
        else:
            # No human pose detected at all
            if not hasattr(self, '_no_pose_warned'):
                self._no_pose_warned = True
                self.get_logger().debug('⚠️ No human pose detected; robot arms hold position')
        
        # Process hand gesture recognition + yaw estimation
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            wrists_px = self._get_pose_wrist_pixels(pose_results.pose_landmarks if pose_results else None, image.shape)
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                # Determine left/right hand (note mirror flip)
                hand_label = handedness.classification[0].label
                # Binding rule: prefer nearest wrist; fallback to MediaPipe label if uncertain
                arm = self._assign_arm_for_hand(hand_landmarks, wrists_px, image.shape, hand_label)
                
                # Recognize gesture
                gesture = self.recognize_hand_gesture(hand_landmarks)
                stable_gesture = self.update_gesture_buffer(arm, gesture)
                
                # Update gripper state based on stable gesture
                if stable_gesture is not None:
                    if arm == 'left':
                        if stable_gesture != self.left_arm_state['gripper_open']:
                            self.left_arm_state['gripper_open'] = stable_gesture
                            self.publish_gripper_command('left', stable_gesture)
                    else:
                        if stable_gesture != self.right_arm_state['gripper_open']:
                            self.right_arm_state['gripper_open'] = stable_gesture
                            self.publish_gripper_command('right', stable_gesture)

                # Estimate yaw and apply smoothing
                yaw_new = self.estimate_wrist_yaw(hand_landmarks)
                if arm == 'left':
                    self.left_arm_state['yaw'] = self.smooth_yaw(yaw_new, self.left_arm_state['prev_yaw'])
                    self.left_arm_state['prev_yaw'] = self.left_arm_state['yaw']
                    # Update yaw component in last_relative
                    dx, dz, _ = self.left_arm_state['last_relative']
                    self.left_arm_state['last_relative'] = (dx, dz, float(self.left_arm_state['yaw']))
                else:
                    self.right_arm_state['yaw'] = self.smooth_yaw(yaw_new, self.right_arm_state['prev_yaw'])
                    self.right_arm_state['prev_yaw'] = self.right_arm_state['yaw']
                    dx, dz, _ = self.right_arm_state['last_relative']
                    self.right_arm_state['last_relative'] = (dx, dz, float(self.right_arm_state['yaw']))
        
        # Draw visualization
        image = self.draw_visualization(image, pose_results.pose_landmarks, hands_results)
        
        # Show image (visual-only scaling)
        display_image = image
        try:
            scale = float(self.display_scale)
        except Exception:
            scale = 1.0
        if scale and abs(scale - 1.0) > 1e-3:
            h, w = image.shape[:2]
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            display_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Teleoperation End-Effector Control', display_image)
        
        # Handle keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Exiting program')
            rclpy.shutdown()
        # 'r' reset removed; use RViz button/topic/service instead
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"teleop_capture_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            self.get_logger().info(f'Saved screenshot: {filename}')
    
    def destroy_node(self):
        """Cleanup resources"""
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
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
