#!/usr/bin/env python3
"""
RViz 急停 按钮（交互标记）

在 RViz 中添加 InteractiveMarkers 显示，并将 Update Topic 设置为：/teleop_estop/update
点击按钮调用 /teleop/estop(Trigger)，触发急停（Teleop 输出禁用）。
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
        # Interactive Marker Server，命名为 teleop_estop
        self.server = InteractiveMarkerServer(self, 'teleop_estop')

        # 参数：允许通过 ROS 参数调节按钮位置与大小
        self.declare_parameter('button_x', -0.2)
        self.declare_parameter('button_y', 0.5)
        self.declare_parameter('button_z', 0.8)
        self.declare_parameter('button_size', 0.1)
        bx = float(self.get_parameter('button_x').value)
        by = float(self.get_parameter('button_y').value)
        bz = float(self.get_parameter('button_z').value)
        bsize = float(self.get_parameter('button_size').value)

        # 创建按钮
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = 'world'
        int_marker.name = 'estop_button'
        int_marker.description = ''  # 去掉自带小标签
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

        # 服务客户端
        self.cli = self.create_client(Trigger, '/teleop/estop')
        self.get_logger().info('RViz E-Stop 按钮已创建，RViz里配置 InteractiveMarkers: /teleop_estop/update')

    def _process_feedback(self, feedback):
        evt = feedback.event_type
        if evt in (1, 5):  # BUTTON_CLICK / MOUSE_UP
            if not self.cli.service_is_ready():
                self.get_logger().warn('等待 /teleop/estop 服务就绪...')
                if not self.cli.wait_for_service(timeout_sec=2.0):
                    self.get_logger().error('服务未就绪，无法急停')
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
                self.get_logger().error(f'急停失败: {resp.message}')
        except Exception as e:
            self.get_logger().error(f'调用服务异常: {e}')


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
