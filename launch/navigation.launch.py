"""Nav2 navigation stack against the live SLAM /map (no map_server / amcl: the map comes from the running SLAM).

    ros2 launch vlm_nav navigation.launch.py                  # camera pipeline (slam.launch.py)
    ros2 launch vlm_nav navigation.launch.py sensor:=lidar    # lidar pipeline  (slam_lidar.launch.py)
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _config_dir():
    try:
        return os.path.join(get_package_share_directory('vlm_nav'), 'config')
    except PackageNotFoundError:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')


def generate_launch_description():
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file': LaunchConfiguration('params_file'),
        }.items())
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('sensor', default_value='camera', description='camera or lidar'),
        # nav2_params_<sensor>.yaml unless params_file is given explicitly
        DeclareLaunchArgument('params_file', default_value=[os.path.join(_config_dir(), 'nav2_params_'),
                                                            LaunchConfiguration('sensor'), '.yaml']),
        nav2,
    ])
