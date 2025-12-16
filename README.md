# Teleoperation_Challenge

本项目用于双臂 UR3e + Robotiq 85 夹爪的末端执行器遥操作与仿真展示，通过 ROS 2、RViz、MoveIt 配置与 Python 节点，实现 3DOF（X/Z/偏航）与手势控制（开/合）的直观演示与控制。

---

### 安装与构建（集成）

1) 准备 ROS 2 Humble：
```bash
# 安装并配置 ROS 2（参考官方文档）
source /opt/ros/humble/setup.bash
```

2) 创建/进入 Python 环境：
```bash
# 使用 Conda
conda create -n env_6dofarms python=3.10 -y
conda activate env_6dofarms
```

3) 安装依赖：
- ROS Python 与可视化（Humble）：
```bash
sudo apt update
sudo apt install -y ros-humble-rclpy rviz2 robot-state-publisher joint-state-publisher
```

- Python 必需依赖：
```bash
pip install numpy opencv-python tflite-runtime pillow pyyaml scipy
```
- 手势/关键点检测依赖：
```bash
pip install mediapipe
```

4) 构建与加载工作空间：
```bash
# 在仓库根目录执行
colcon build
source install/setup.bash
```

提示：如仓库已包含 `build/` 与 `install/`，仍建议在本机重新 `colcon build` 以适配本地环境。

5) 开源依赖与工具：
    - ROS 2 社区包（`robot_state_publisher`, `rviz2`, `joint_state_publisher_gui` 等）
    - UR 官方/社区描述与 MoveIt 配置包
    - Robotiq 85 社区驱动与消息包
    - 引用模型/驱动来源：
      - Robotiq 85 夹爪模型与驱动：`robotiq_85_gripper`（来源：Robotiq/社区开源资源）
      - UR 机械臂描述：Universal Robots ROS 2 Description（`Universal_Robots_ROS2_Description`）
      - UR 机械臂驱动：Universal Robots ROS 2 Driver（`Universal_Robots_ROS2_Driver`）

  注：上述资源分别来自其官方或社区维护的开源仓库，具体版本与许可证以对应仓库为准。

## 快速开始

在三个终端中分别运行如下命令：

- 终端 1：启动模型展示（RViz）

```bash
ros2 launch dual_ur3e_description dual_ur3e_display.launch.py
```

- 终端 2：启动末端执行器命令发布器（状态打印与演示）

```bash
python3 scripts/endeffector_commander.py
```

- 终端 3：启动遥操作控制（3DOF + 手势）

```bash
python3 scripts/teleop_endeffector_control.py
```

运行后，终端会打印：
- 工作空间范围与控制映射（X/Z/偏航 ←→ 手腕位姿）
- 手势映射（张开手掌 → 打开夹爪；握拳 → 关闭夹爪）
- 键盘快捷键：`q` 退出、`r` 重置初始位置、`s` 保存截图
- 可见性提示：当人体关键点或手势暂不可见时，保持当前位置不动

---

## 项目结构（简要）

- `scripts/`
  - `endeffector_commander.py`: 末端执行器指令发布与演示、关节状态打印
  - `teleop_endeffector_control.py`: 遥操作主控，映射腕部姿态到末端位置与偏航，同时处理手势控制
- `dual_ur3e_description/`: 双臂与夹爪的模型与显示 Launch
- `robotiq_85_*`: Robotiq 85 相关包
- `ur_*`: UR 相关描述、控制与 MoveIt 配置
- `docs/USAGE_GUIDE.md`: 额外使用指南（如有）
- `build/`, `install/`, `log/`: 由 `colcon` 构建生成

---

---

## 运行前检查

- 已正确 `source` ROS 2 环境（如 `/opt/ros/<distro>/setup.bash`）与本工作空间 `install/setup.bash`
- 摄像头/输入设备工作正常，OpenCV 能够访问（若使用手势实时识别）
- 必要话题/TF 发布正常。可用如下命令做简单检查：
```bash
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic hz /joint_states
```

---

## 许可证

- 本仓库中的外部包（UR、Robotiq、ROS 社区包）遵循其各自许可证
- 本项目代码请依据仓库许可证文件（若未声明，可补充 MIT/Apache-2.0 等）

---
### 开源引用说明
- Robotiq 85 夹爪模型与驱动：`robotiq_85_gripper`（参考社区/官方开源实现）
- Universal Robots ROS 2 Description：`Universal_Robots_ROS2_Description`
- Universal Robots ROS 2 Driver：`Universal_Robots_ROS2_Driver`

以上资源版权与许可证归其原作者与维护方所有，使用与分发请遵循各仓库的 LICENSE。

---
## 致谢

- ROS 2 与开源机器人社区
- UR 与 Robotiq 相关开源资源维护者

如需将依赖精确固化到 `requirements.txt` 或补充真实硬件运行指南，请告知，我将补充相应文档与脚本。