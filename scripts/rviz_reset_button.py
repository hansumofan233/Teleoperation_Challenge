#!/usr/bin/env python3
"""
RViz 重置按钮（交互标记）

在 RViz 中添加 InteractiveMarkers 显示，并将 Update Topic 设置为：/teleop_reset/update
点击按钮即可调用 /teleop/reset_reference 服务，重置 Teleop 参考与模型初始位置。
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
        # Interactive Marker Server，命名为 teleop_reset
        self.server = InteractiveMarkerServer(self, 'teleop_reset')

        # 参数：允许通过 ROS 参数调节按钮位置与大小
        self.declare_parameter('button_x', 0.0)
        self.declare_parameter('button_y', 0.5)
        self.declare_parameter('button_z', 1.5)
        self.declare_parameter('button_size', 0.1)  
        bx = float(self.get_parameter('button_x').value)
        by = float(self.get_parameter('button_y').value)
        bz = float(self.get_parameter('button_z').value)
        bsize = float(self.get_parameter('button_size').value)

        # 创建一个按钮样式的交互标记
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = 'world'
        int_marker.name = 'reset_button'
        # 去掉交互标记自带描述，避免出现两个“Reset”标签
        int_marker.description = ''
        int_marker.scale = max(0.1, bsize)
        int_marker.pose = Pose()
        int_marker.pose.position.x = 0.3
        int_marker.pose.position.y = 0.25
        int_marker.pose.position.z = 0.8

        # 可视化方块
        box = Marker()
        box.type = Marker.CUBE
        box.scale.x = bsize
        box.scale.y = bsize
        box.scale.z = max(0.04, bsize * 0.45)
        box.color.r = 1.0
        box.color.g = 0.2
        box.color.b = 0.2
        box.color.a = 0.9

        # 文本标签
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

        # 固定按钮位置：不添加拖拽控件，避免误移动

        # ROS 2 API: 先插入，再通过 setCallback 绑定反馈
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self._process_feedback)
        self.server.applyChanges()

        # 准备服务客户端 + 重置广播话题
        self.cli = self.create_client(Trigger, '/teleop/reset_reference')
        self.reset_pub = self.create_publisher(Empty, '/teleop/reset', 10)
        self.get_logger().info('RViz Reset 按钮已创建，RViz里配置 InteractiveMarkers: /teleop_reset/update')

    def _process_feedback(self, feedback):
        # 响应按钮点击：兼容 BUTTON_CLICK(1) 与 MOUSE_UP(5)
        evt = feedback.event_type
        if evt in (1, 5):
            if not self.cli.service_is_ready():
                self.get_logger().warn('等待 /teleop/reset_reference 服务就绪...')
                if not self.cli.wait_for_service(timeout_sec=2.0):
                    self.get_logger().error('服务未就绪，无法重置')
                    return
            # 先调用 Teleop 主服务完成模型重置
            req = Trigger.Request()
            future = self.cli.call_async(req)
            future.add_done_callback(self._on_call_done)
            # 并向其他节点广播统一重置信号（如摄像头节点）
            self.reset_pub.publish(Empty())

    def _on_call_done(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info(resp.message)
            else:
                self.get_logger().error(f'重置失败: {resp.message}')
        except Exception as e:
            self.get_logger().error(f'调用服务异常: {e}')


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
