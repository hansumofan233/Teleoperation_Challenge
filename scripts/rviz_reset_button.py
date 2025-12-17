#!/usr/bin/env python3
"""
RViz Reset button (Interactive Marker)

Add an InteractiveMarkers display in RViz and set Update Topic to: /teleop_reset/update
Clicking the button calls /teleop/reset_reference to reset teleop reference and model initial pose.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Empty

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from geometry_msgs.msg import Pose


class RVizResetButton(Node):
    def __init__(self):
        super().__init__('rviz_reset_button')
        # Interactive Marker Server named teleop_reset
        self.server = InteractiveMarkerServer(self, 'teleop_reset')

        # Parameters: tune button position and size via ROS parameters
        self.declare_parameter('button_x', 0.0)
        self.declare_parameter('button_y', 0.5)
        self.declare_parameter('button_z', 1.5)
        self.declare_parameter('button_size', 0.1)  
        bx = float(self.get_parameter('button_x').value)
        by = float(self.get_parameter('button_y').value)
        bz = float(self.get_parameter('button_z').value)
        bsize = float(self.get_parameter('button_size').value)

        # Create a button-style interactive marker
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = 'world'
        int_marker.name = 'reset_button'
        int_marker.description = ''  # remove default description to avoid duplicate "Reset" labels
        int_marker.scale = max(0.1, bsize)
        int_marker.pose = Pose()
        int_marker.pose.position.x = 0.3
        int_marker.pose.position.y = 0.25
        int_marker.pose.position.z = 0.8

        # Visual cube
        box = Marker()
        box.type = Marker.CUBE
        box.scale.x = bsize
        box.scale.y = bsize
        box.scale.z = max(0.04, bsize * 0.45)
        box.color.r = 1.0
        box.color.g = 0.2
        box.color.b = 0.2
        box.color.a = 0.9

        # Text label
        text = Marker()
        text.type = Marker.TEXT_VIEW_FACING
        text.text = 'Reset'
        text.scale.z = max(0.035, bsize * 0.36)
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.pose.position.z = 0.12

        control_btn = InteractiveMarkerControl()
        control_btn.always_visible = True
        control_btn.interaction_mode = InteractiveMarkerControl.BUTTON
        control_btn.markers.append(box)
        control_btn.markers.append(text)
        int_marker.controls.append(control_btn)

        # Fixed button position: do not add drag controls to prevent accidental moves

        # ROS 2 API: insert then bind feedback with setCallback
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self._process_feedback)
        self.server.applyChanges()

        # Prepare service client + reset broadcast topic
        self.cli = self.create_client(Trigger, '/teleop/reset_reference')
        self.reset_pub = self.create_publisher(Empty, '/teleop/reset', 10)
        self.get_logger().info('RViz Reset button created. In RViz, configure InteractiveMarkers: /teleop_reset/update')

    def _process_feedback(self, feedback):
        # Handle button click: support BUTTON_CLICK(1) and MOUSE_UP(5)
        evt = feedback.event_type
        if evt in (1, 5):
            if not self.cli.service_is_ready():
                self.get_logger().warn('Waiting for /teleop/reset_reference service to be ready...')
                if not self.cli.wait_for_service(timeout_sec=2.0):
                    self.get_logger().error('Service not ready, cannot reset')
                    return
            # First call teleop main service to reset model
            req = Trigger.Request()
            future = self.cli.call_async(req)
            future.add_done_callback(self._on_call_done)
            # Then broadcast a unified reset signal to other nodes (e.g., camera)
            self.reset_pub.publish(Empty())

    def _on_call_done(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info(resp.message)
            else:
                self.get_logger().error(f'Reset failed: {resp.message}')
        except Exception as e:
            self.get_logger().error(f'Exception calling service: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RVizResetButton()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
