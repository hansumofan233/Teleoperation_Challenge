#!/usr/bin/env python3
"""
RViz 启用/停用 按钮（交互标记）

在 RViz 中添加 InteractiveMarkers 显示，并将 Update Topic 设置为：/teleop_enable/update
点击按钮在 启用 与 停用 之间切换，对应调用 /teleop/enable(SetBool)。
注意：本按钮内部维护本地切换状态，未从服务端读取真实状态，仅用于单独功能验证。
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from geometry_msgs.msg import Pose


class RVizEnableButton(Node):
    def __init__(self):
        super().__init__('rviz_enable_button')
        # Interactive Marker Server，命名为 teleop_enable
        self.server = InteractiveMarkerServer(self, 'teleop_enable')

        # 参数：允许通过 ROS 参数调节按钮位置与大小
        self.declare_parameter('button_x', 0.2)
        self.declare_parameter('button_y', 0.5)
        self.declare_parameter('button_z', 0.8)
        self.declare_parameter('button_size', 0.1)
        self.bx = float(self.get_parameter('button_x').value)
        self.by = float(self.get_parameter('button_y').value)
        self.bz = float(self.get_parameter('button_z').value)
        self.bsize = float(self.get_parameter('button_size').value)

        # 本地启用标志（仅用于切换显示与请求数据）
        self.enabled = False  # 初始为未启用，第一次点击执行启用

        # 服务客户端
        self.cli = self.create_client(SetBool, '/teleop/enable')

        # 创建按钮
        self._create_or_update_button()
        self.get_logger().info('RViz Enable/Disable 按钮已创建，RViz里配置 InteractiveMarkers: /teleop_enable/update')

    def _build_marker(self, label: str, color_rgb):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = 'world'
        int_marker.name = 'enable_button'
        int_marker.description = ''  # 去掉自带小标签
        int_marker.scale = max(0.1, self.bsize)
        int_marker.pose = Pose()
        int_marker.pose.position.x = 0.8
        int_marker.pose.position.y = 0.25
        int_marker.pose.position.z = 0.8

        box = Marker()
        box.type = Marker.CUBE
        box.scale.x = self.bsize
        box.scale.y = self.bsize
        box.scale.z = max(0.04, self.bsize * 0.45)
        box.color.r = float(color_rgb[0])
        box.color.g = float(color_rgb[1])
        box.color.b = float(color_rgb[2])
        box.color.a = 0.9

        text = Marker()
        text.type = Marker.TEXT_VIEW_FACING
        text.text = label
        text.scale.z = max(0.035, self.bsize * 0.36)
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
        return int_marker

    def _create_or_update_button(self):
        # 根据当前状态决定下次动作的标签与颜色
        # 未启用 -> 显示 Enable（绿色）；已启用 -> 显示 Disable（橙色）
        if not self.enabled:
            label = 'Enable'
            color = (0.2, 0.8, 0.2)
        else:
            label = 'Disable'
            color = (1.0, 0.6, 0.1)

        int_marker = self._build_marker(label, color)

        # 如果已存在则更新
        try:
            self.server.erase(int_marker.name)
        except Exception:
            pass
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self._process_feedback)
        self.server.applyChanges()

    def _process_feedback(self, feedback):
        evt = feedback.event_type
        if evt not in (1, 5):  # BUTTON_CLICK / MOUSE_UP
            return
        # 启动服务客户端
        if not self.cli.service_is_ready():
            self.get_logger().warn('等待 /teleop/enable 服务就绪...')
            if not self.cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().error('服务未就绪，无法切换启停')
                return

        # 目标状态：当前未启用->启用(True)，已启用->停用(False)
        target = not self.enabled
        req = SetBool.Request()
        req.data = bool(target)
        future = self.cli.call_async(req)
        future.add_done_callback(self._on_call_done)

    def _on_call_done(self, future):
        try:
            resp = future.result()
            if resp.success:
                # 本地状态翻转，并刷新按钮显示
                self.enabled = not self.enabled
                self._create_or_update_button()
                self.get_logger().info(resp.message)
            else:
                self.get_logger().error(f'切换失败: {resp.message}')
        except Exception as e:
            self.get_logger().error(f'调用服务异常: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RVizEnableButton()
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
