# 末端执行器摄像头遥操作系统

## 系统概述

该系统使用摄像头捕获操作员的手臂运动,实现对双臂机器人末端执行器的遥操作控制。

### 控制模式
- **2D平面运动**: 末端执行器在XY平面内移动
- **1D旋转**: 末端执行器绕Z轴旋转
- **双臂独立控制**: 左右臂可独立控制

## 文件说明

### 核心节点

1. **teleop_endeffector_control.py**
   - 摄像头采集和姿态检测
   - 手臂运动到机器人坐标映射
   - 末端执行器目标位置发布
   - 实时可视化界面

2. **endeffector_commander.py**
   - 接收末端执行器目标位置
   - 逆运动学求解
   - 生成关节轨迹命令
   - 平滑运动控制

3. **teleop_tester.py**
   - 系统测试和调试工具
   - 消息监听和统计
   - 性能分析

### Launch文件

- **teleop_endeffector.launch.py**
  - 一键启动完整系统
  - 包含机器人模型、RViz、控制节点

## 使用说明

### 1. 安装依赖

```bash
# 安装Python依赖
pip install opencv-python mediapipe numpy scipy

# 确保ROS2工作空间已配置
cd ~/Teleoperation_Challenge
source install/setup.bash
```

### 2. 给脚本添加执行权限

```bash
chmod +x scripts/teleop_endeffector_control.py
chmod +x scripts/endeffector_commander.py
chmod +x scripts/teleop_tester.py
```

### 3. 启动系统

```bash
# 启动完整遥操作系统
ros2 launch dual_ur3e_description teleop_endeffector.launch.py

# 可选:自定义参数
ros2 launch dual_ur3e_description teleop_endeffector.launch.py \
    camera_id:=0 \
    workspace_x_min:=0.2 \
    workspace_x_max:=0.8 \
    workspace_y_min:=-0.3 \
    workspace_y_max:=0.3 \
    workspace_z:=0.9
```

### 4. 系统校准

启动后会打开摄像头窗口:

1. **初始姿态**: 站在摄像头前,双臂自然下垂或T字型展开
2. **按'c'键校准**: 将当前姿态设为参考位置
3. **开始控制**: 移动手臂,机器人末端执行器会跟随运动

### 5. 控制按键

- **q**: 退出程序
- **c**: 校准(记录当前姿态为参考)
- **r**: 重置校准
- **s**: 保存当前截图

## 运动映射说明

### 坐标映射

| 人体运动 | 机器人运动 | 说明 |
|---------|-----------|------|
| 手臂前后移动 | X轴移动 | 前伸增加X,后收减小X |
| 手臂左右移动 | Y轴移动 | 左右对应Y轴变化 |
| 手臂上下移动 | (固定Z) | Z轴保持固定高度 |
| 前臂旋转角度 | 绕Z轴旋转 | 前臂方向映射到末端旋转 |

### 工作空间

默认工作空间范围:
- X轴: 0.2m - 0.8m (前后)
- Y轴: -0.3m - 0.3m (左右)
- Z轴: 0.9m (固定高度)

可通过launch参数调整。

## 调试工具

### 测试节点

在新终端中运行测试工具:

```bash
source install/setup.bash
ros2 run dual_ur3e_teleop teleop_tester.py
```

这会显示:
- 消息接收频率
- 末端执行器目标位置
- 关节命令状态

### 话题监控

```bash
# 查看末端执行器目标
ros2 topic echo /left_arm/endeffector_target
ros2 topic echo /right_arm/endeffector_target

# 查看关节命令
ros2 topic echo /left_arm/joint_trajectory_controller/joint_trajectory
ros2 topic echo /right_arm/joint_trajectory_controller/joint_trajectory
```

## 参数调优

### 位置平滑度
```bash
ros2 launch dual_ur3e_description teleop_endeffector.launch.py \
    position_smoothing:=0.5  # 0-1, 值越大越平滑但响应慢
```

### 旋转灵敏度
```bash
ros2 launch dual_ur3e_description teleop_endeffector.launch.py \
    rotation_sensitivity:=3.0  # 增加旋转响应
```

### 工作空间
```bash
ros2 launch dual_ur3e_description teleop_endeffector.launch.py \
    workspace_x_min:=0.3 \
    workspace_x_max:=0.7 \
    workspace_y_min:=-0.2 \
    workspace_y_max:=0.2
```

## 常见问题

### 1. 摄像头无法打开
- 检查摄像头连接
- 尝试不同camera_id: `camera_id:=1`

### 2. 检测不到手臂
- 确保光线充足
- 调整摄像头位置,全身或上半身入镜
- 穿着对比度高的衣服

### 3. 运动不流畅
- 增加position_smoothing值
- 检查系统性能(CPU/GPU负载)
- 降低摄像头分辨率

### 4. 末端执行器不动
- 检查ROS话题是否正常发布
- 运行teleop_tester.py查看消息
- 查看RViz中机器人模型状态

## 系统架构

```
摄像头
  ↓
MediaPipe姿态检测
  ↓
手臂坐标提取
  ↓
坐标映射+平滑滤波
  ↓
末端执行器目标 (/endeffector_target)
  ↓
逆运动学求解
  ↓
关节轨迹命令 (/joint_trajectory)
  ↓
机器人控制器
  ↓
机器人运动
```

## 下一步改进

1. **增加力反馈**: 集成触觉设备
2. **双手精细控制**: 增加手指姿态识别
3. **自适应工作空间**: 根据用户身高自动调整
4. **多用户支持**: 识别并跟踪特定操作员
5. **记录与回放**: 保存操作序列用于示教

## 技术支持

如有问题,请检查:
1. ROS2日志: `ros2 topic list`, `ros2 node list`
2. 摄像头窗口显示的状态信息
3. 终端输出的错误信息
