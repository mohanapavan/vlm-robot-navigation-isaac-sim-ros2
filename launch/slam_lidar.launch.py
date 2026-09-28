"""Lidar SLAM pipeline: slam_toolbox on the simulated 2-D lidar (/scan). One launch instead of the separate terminals.

    ros2 launch vlm_nav slam_lidar.launch.py                          # build a new map (mapping mode)
    ros2 launch vlm_nav slam_lidar.launch.py mode:=localization      # resume from the saved map (see below)
    ros2 launch vlm_nav slam_lidar.launch.py mode:=static            # saved map only, no slam_toolbox (see below)

Resume: after mapping, save the SLAM state (docs/commands.md, section 4):
    ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '/home/user/my_map_posegraph'}"
Later, restart the simulator (the robot returns to its start pose, which is the map origin) and launch with
mode:=localization: the same map frame is restored, so scene-graph coordinates stay valid.

In localization mode Nav2 gets its /map from nav2_map_server serving the saved map image (map_yaml:=, default
~/my_map.yaml, written by map_saver_cli), while slam_toolbox only provides the robot's localization (its own map is
remapped to /slam_toolbox/map). slam_toolbox re-publishes a slightly different-sized map every so often, and every
size change resets Nav2's costmaps, which showed up as aborted goals in the middle of a drive.

mode:=static needs only the saved map image (map_yaml:=, default ~/my_map.yaml), not the slam_toolbox pose graph: it
serves that map to Nav2 and publishes an identity map -> odom transform. That is exact as long as the robot starts at
the pose the map was built from (the simulator spawns it at the map origin after Stop -> Play) and odometry does not
drift, which holds in Isaac Sim. Checked against the live /scan by scripts/check_map_alignment.py.

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
import sys

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _config_dir():
    try:
        return os.path.join(get_package_share_directory('vlm_nav'), 'config')
    except PackageNotFoundError:  # running straight from a source checkout
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')


def _saved_map_nodes(context, *args, **kwargs):
    """map_server + its lifecycle manager for `localization` / `static` mode, serving a map whose unexplored space is unknown."""
    if LaunchConfiguration('mode').perform(context) not in ('localization', 'static'):
        return []
    try:
        from vlm_nav.map_grid import corrected_map_yaml
    except ImportError:  # running straight from a source checkout
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from vlm_nav.map_grid import corrected_map_yaml
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    yaml_file = corrected_map_yaml(os.path.expanduser(LaunchConfiguration('map_yaml').perform(context)))
    return [
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen',
             parameters=[{'yaml_filename': yaml_file}, sim_time]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_map',
             output='screen', parameters=[{'autostart': True, 'node_names': ['map_server']}, sim_time]),
    ]


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
    in_localization = IfCondition(PythonExpression(["'", mode, "' == 'localization'"]))
    in_static = IfCondition(PythonExpression(["'", mode, "' == 'static'"]))
    localization = Node(
        package='slam_toolbox', executable='localization_slam_toolbox_node', name='slam_toolbox',
        parameters=[slam_params, sim_time,
                    {'mode': 'localization', 'map_file_name': LaunchConfiguration('map_file')}],
        remappings=[('/map', '/slam_toolbox/map')],
        output='screen', condition=in_localization)
    static_map_odom = Node(
        package='tf2_ros', executable='static_transform_publisher', name='map_odom_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[sim_time], condition=in_static)

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', os.path.join(cfg, 'map_view.rviz')],
                parameters=[sim_time], condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('lidar_frame', default_value='front_2d_lidar',
                              description='TF frame the scan is physically attached to (scan frame_id is base_scan)'),
        DeclareLaunchArgument('mode', default_value='mapping',
                              description='mapping, localization (saved pose graph) or static (saved map only)'),
        DeclareLaunchArgument('map_file', default_value=os.path.expanduser('~/my_map_posegraph'),
                              description='saved slam_toolbox pose graph, without extension (localization mode)'),
        DeclareLaunchArgument('map_yaml', default_value=os.path.expanduser('~/my_map.yaml'),
                              description='saved map image (localization / static mode): served to Nav2 as /map'),
        DeclareLaunchArgument('camera', default_value='front_stereo_camera'),
        DeclareLaunchArgument('rviz', default_value='true'),
        lidar_tf, optical_tf('left'), optical_tf('right'), mapping, localization, OpaqueFunction(function=_saved_map_nodes),
        static_map_odom, rviz,
    ])
