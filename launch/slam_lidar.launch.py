"""Lidar SLAM pipeline: slam_toolbox on the simulated 2-D lidar (/scan). One launch instead of the separate terminals.

    ros2 launch vlm_nav slam_lidar.launch.py                          # build a new map (mapping mode)
    ros2 launch vlm_nav slam_lidar.launch.py mode:=localization      # resume from the saved map (see below)

Resume: after mapping, save the SLAM state (docs/commands.md, section 4):
    ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '/home/user/my_map_posegraph'}"
Later, restart the simulator (the robot returns to its start pose, which is the map origin) and launch with
mode:=localization: the same map frame is restored, so scene-graph coordinates stay valid.

Starts, with use_sim_time:
  - a static transform  front_2d_lidar -> base_scan  (the scan's frame_id is `base_scan`, but the simulator only
    publishes TF for `front_2d_lidar`, which sits at the same place with the same orientation)
  - the two camera optical-frame transforms (image headers use *_optical, the simulator only publishes *_rgb), so
    the scene-graph builder can find the camera in TF
  - slam_toolbox (async, mapping): publishes /map and the map -> odom TF
  - rviz2 (optional)

The simulator already publishes odom -> base_link, so there is no EKF and no map_relay here (slam_toolbox publishes
a latched /map itself). Do not run this together with slam.launch.py: both would publish map -> odom.

Use it with:  ros2 launch vlm_nav navigation.launch.py sensor:=lidar
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _config_dir():
    try:
        return os.path.join(get_package_share_directory('vlm_nav'), 'config')
    except PackageNotFoundError:  # running straight from a source checkout
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')


def generate_launch_description():
    cfg = _config_dir()
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    lidar_tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='base_scan_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--frame-id', LaunchConfiguration('lidar_frame'), '--child-frame-id', 'base_scan'],
        parameters=[sim_time])

    def optical_tf(side):
        return Node(
            package='tf2_ros', executable='static_transform_publisher', name=f'{side}_optical_tf',
            arguments=['--x', '0', '--y', '0', '--z', '0',
                       '--frame-id', [LaunchConfiguration('camera'), f'_{side}_rgb'],
                       '--child-frame-id', [LaunchConfiguration('camera'), f'_{side}_optical']],
            parameters=[sim_time])

    mode = LaunchConfiguration('mode')
    slam_params = os.path.join(cfg, 'slam_toolbox_lidar.yaml')
    mapping = Node(
        package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
        parameters=[slam_params, sim_time], output='screen',
        condition=IfCondition(PythonExpression(["'", mode, "' == 'mapping'"])))
    localization = Node(
        package='slam_toolbox', executable='localization_slam_toolbox_node', name='slam_toolbox',
        parameters=[slam_params, sim_time,
                    {'mode': 'localization', 'map_file_name': LaunchConfiguration('map_file')}],
        output='screen', condition=IfCondition(PythonExpression(["'", mode, "' == 'localization'"])))

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', os.path.join(cfg, 'map_view.rviz')],
                parameters=[sim_time], condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('lidar_frame', default_value='front_2d_lidar',
                              description='TF frame the scan is physically attached to (scan frame_id is base_scan)'),
        DeclareLaunchArgument('mode', default_value='mapping', description='mapping or localization'),
        DeclareLaunchArgument('map_file', default_value=os.path.expanduser('~/my_map_posegraph'),
                              description='saved slam_toolbox pose graph, without extension (localization mode)'),
        DeclareLaunchArgument('camera', default_value='front_stereo_camera'),
        DeclareLaunchArgument('rviz', default_value='true'),
        lidar_tf, optical_tf('left'), optical_tf('right'), mapping, localization, rviz,
    ])
