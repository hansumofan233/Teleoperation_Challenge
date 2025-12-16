#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # 获取包路径
    pkg_dir = get_package_share_directory('dual_ur3e_description')
    
    # Launch参数
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gui = LaunchConfiguration('use_gui')
    
    # 文件路径
    urdf_file = os.path.join(pkg_dir, 'urdf', 'dual_ur3e.urdf.xacro')
    rviz_config = os.path.join(pkg_dir, 'config', 'dual_ur3e.rviz')
    
    # 初始关节位置
    initial_joint_positions = {
        'left_wrist_1_joint': -1.5708,
        'left_wrist_2_joint': 0.0,
        'left_wrist_3_joint': -3.1416,
        'right_wrist_1_joint': -1.5708,
        'right_wrist_2_joint': 0.0,
        'right_wrist_3_joint': 3.1416,
    }
    
    return LaunchDescription([
        # Launch参数声明
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),
        
        DeclareLaunchArgument(
            'use_gui',
            default_value='false',
            description='Start joint_state_publisher_gui (set to false for teleoperation)'
        ),
        
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': ParameterValue(Command(['xacro ', urdf_file]), value_type=str),
                'use_sim_time': use_sim_time
            }]
        ),
        
        # Joint State Publisher (GUI) - only if use_gui is true
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            condition=IfCondition(use_gui)
        ),

        # Joint State Publisher (non-GUI) - only if use_gui is false
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time},
                        initial_joint_positions],
            condition=UnlessCondition(use_gui)
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}]
        )
    ])
