"""SLAM pipeline: optical-frame TFs, EKF, RTAB-Map (stereo) and the /map relay, in one launch.

Replaces the separate terminals for both static transform publishers, ekf_node, the rtabmap
launch and map_relay.py. Isaac Sim must already be playing and publishing /clock.

    ros2 launch vlm_nav slam.launch.py                   # continue/extend ~/rtabmap.db if it exists
    ros2 launch vlm_nav slam.launch.py fresh_map:=true   # delete the database on start (-d)
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


# Occupancy-grid settings for a low, forward-looking stereo camera on a flat floor. With RTAB-Map's defaults the
# grid built from stereo alone is ~90% unknown with the robot's own cell marked occupied, so Nav2 can never plan:
# ray tracing carves free space between the camera and each obstacle, normals-based ground segmentation is
# replaced by a plain height threshold (the textureless floor gives too few points for normals), and obstacles
# above the robot (racks, ceiling) or far away (noisy stereo depth) are ignored.
GRID_ARGS = ' '.join([
    '--Grid/RayTracing true',
    '--Grid/3D false',
    '--Grid/NormalsSegmentation false',
    '--Grid/MaxGroundHeight 0.05',
    '--Grid/MaxObstacleHeight 1.5',
    '--Grid/RangeMax 6.0',
    '--Grid/NoiseFilteringRadius 0.1',
    '--Grid/NoiseFilteringMinNeighbors 4',
])


def _config_dir():
    try:
        return os.path.join(get_package_share_directory('vlm_nav'), 'config')
    except PackageNotFoundError:  # running straight from a source checkout
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')


def generate_launch_description():
    cfg = _config_dir()
    use_sim_time = LaunchConfiguration('use_sim_time')
    cam = LaunchConfiguration('camera')          # e.g. front_stereo_camera
    sim_time = {'use_sim_time': use_sim_time}

    def optical_tf(side):
        # Isaac Sim's camera_info/image headers use *_optical frames but only publish TF for *_rgb
        # (which already follows the optical convention), so the two are identical.
        return Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'{side}_optical_tf',
            arguments=['--x', '0', '--y', '0', '--z', '0',
                       '--frame-id', [cam, f'_{side}_rgb'],
                       '--child-frame-id', [cam, f'_{side}_optical']],
            parameters=[sim_time])

    ekf = Node(
        package='robot_localization', executable='ekf_node', name='ekf_filter_node',
        parameters=[os.path.join(cfg, 'ekf.yaml'), sim_time], output='screen')

    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py')),
        launch_arguments={
            'stereo': 'true',
            'use_sim_time': use_sim_time,
            'frame_id': 'base_link',
            'visual_odometry': 'false',
            'odom_topic': '/odometry/filtered',
            'imu_topic': LaunchConfiguration('imu_topic'),
            'left_image_topic': ['/', cam, '/left/image_raw'],
            'right_image_topic': ['/', cam, '/right/image_raw'],
            'left_camera_info_topic': ['/', cam, '/left/camera_info'],
            'right_camera_info_topic': ['/', cam, '/right/camera_info'],
            'approx_sync': 'true',
            'rtabmap_viz': 'false',
            'rviz': 'false',
            'database_path': LaunchConfiguration('database_path'),
            # -d deletes the database on start: only when explicitly requested, never by default.
            'args': PythonExpression(["('-d ' if '", LaunchConfiguration('fresh_map'), "' == 'true' else '') + '",
                                      LaunchConfiguration('rtabmap_args'), "'"]),
        }.items())

    map_relay = Node(package='vlm_nav', executable='map_relay', name='map_relay',
                     parameters=[sim_time], output='screen')

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', os.path.join(cfg, 'map_view.rviz')],
                parameters=[sim_time], condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('camera', default_value='front_stereo_camera',
                              description='Stereo camera name prefix used in topics and frames'),
        DeclareLaunchArgument('imu_topic', default_value='/front_stereo_imu/imu'),
        DeclareLaunchArgument('database_path', default_value=os.path.expanduser('~/rtabmap.db')),
        DeclareLaunchArgument('fresh_map', default_value='false',
                              description='true = delete the RTAB-Map database on start'),
        DeclareLaunchArgument('rtabmap_args', default_value=GRID_ARGS,
                              description='extra RTAB-Map flags/parameters (default: stereo occupancy-grid tuning)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        optical_tf('left'), optical_tf('right'), ekf, rtabmap, map_relay, rviz,
    ])
