#!/usr/bin/env python3
"""
RViz E-Stop button (Interactive Marker)

Add an InteractiveMarkers display in RViz and set Update Topic to: /teleop_estop/update
Clicking the button calls /teleop/estop (Trigger) to activate emergency stop (teleop output disabled).
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from geometry_msgs.msg import Pose


class RVizEStopButton(Node):
    def __init__(self):
        super().__init__('rviz_estop_button')
        # Interactive Marker Server named teleop_estop
        self.server = InteractiveMarkerServer(self, 'teleop_estop')

        # Parameters: tune button position and size via ROS parameters
        self.declare_parameter('button_x', -0.2)
        self.declare_parameter('button_y', 0.5)
        self.declare_parameter('button_z', 0.8)
        self.declare_parameter('button_size', 0.1)
        bx = float(self.get_parameter('button_x').value)
        by = float(self.get_parameter('button_y').value)
        bz = float(self.get_parameter('button_z').value)
        bsize = float(self.get_parameter('button_size').value)

        # Create button
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = 'world'
        int_marker.name = 'estop_button'
        int_marker.description = ''  # remove default label
        int_marker.scale = max(0.1, bsize)
        int_marker.pose = Pose()
        int_marker.pose.position.x = 0.55
        int_marker.pose.position.y = 0.25
        int_marker.pose.position.z = 0.8

        box = Marker()
        box.type = Marker.CUBE
        box.scale.x = bsize
        box.scale.y = bsize
        box.scale.z = max(0.04, bsize * 0.45)
        box.color.r = 1.0
        box.color.g = 0.2
        box.color.b = 0.2
        box.color.a = 0.95

        text = Marker()
        text.type = Marker.TEXT_VIEW_FACING
        text.text = 'E-STOP'
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

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self._process_feedback)
        self.server.applyChanges()

        # Service client
        self.cli = self.create_client(Trigger, '/teleop/estop')
        self.get_logger().info('RViz E-Stop button created. In RViz, configure InteractiveMarkers: /teleop_estop/update')

    def _process_feedback(self, feedback):
        evt = feedback.event_type
        if evt in (1, 5):  # BUTTON_CLICK / MOUSE_UP
            if not self.cli.service_is_ready():
                self.get_logger().warn('Waiting for /teleop/estop service to be ready...')
                if not self.cli.wait_for_service(timeout_sec=2.0):
                    self.get_logger().error('Service not ready, cannot perform E-Stop')
                    return
            req = Trigger.Request()
            future = self.cli.call_async(req)
            future.add_done_callback(self._on_call_done)

    def _on_call_done(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().error(resp.message)
            else:
                self.get_logger().error(f'E-Stop failed: {resp.message}')
        except Exception as e:
            self.get_logger().error(f'Exception calling service: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RVizEStopButton()
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
