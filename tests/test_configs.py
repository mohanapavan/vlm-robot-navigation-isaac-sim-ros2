"""Sanity checks for the shipped YAML configs (no ROS needed)."""
from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')
CONFIG = Path(__file__).resolve().parents[1] / 'config'


def _load(name):
    return yaml.safe_load((CONFIG / name).read_text())


@pytest.mark.parametrize('name', ['nav2_params_camera.yaml', 'nav2_params_lidar.yaml'])
@pytest.mark.parametrize('costmap', ['local_costmap', 'global_costmap'])
def test_no_scan_obstacle_layer_in_either_costmap(name, costmap):
    """An ObstacleLayer on /scan broke the costmap's TF handling after start-up (the global costmap's robot pose froze
    at the start-up position, so plans began there; the local one kept a single map->odom sample). Keep it out."""
    params = _load(name)[costmap][costmap]['ros__parameters']
    assert 'obstacle_layer' not in params['plugins'] and 'obstacle_layer' not in params


def test_lidar_params_differ_from_camera_only_where_intended():
    cam, lid = _load('nav2_params_camera.yaml'), _load('nav2_params_lidar.yaml')
    assert lid['bt_navigator']['ros__parameters']['odom_topic'] == '/chassis/odom'
    assert cam['bt_navigator']['ros__parameters']['odom_topic'] == '/odometry/filtered'
    assert lid['global_costmap']['global_costmap']['ros__parameters']['plugins'] == ['static_layer', 'inflation_layer']
    assert lid['local_costmap']['local_costmap']['ros__parameters']['plugins'] == ['static_layer', 'inflation_layer']
    for key in ('controller_server', 'planner_server', 'behavior_server', 'velocity_smoother'):
        assert lid[key] == cam[key], key


def test_slam_toolbox_params_are_correctly_typed():
    p = _load('slam_toolbox_lidar.yaml')['slam_toolbox']['ros__parameters']
    assert p['use_sim_time'] is True and p['scan_topic'] == '/scan' and p['mode'] == 'mapping'
    assert (p['odom_frame'], p['map_frame'], p['base_frame']) == ('odom', 'map', 'base_link')
    assert isinstance(p['min_laser_range'], float) and isinstance(p['max_laser_range'], float)
    assert p['min_laser_range'] > 0.4          # the simulated lidar's real minimum is 0.4000000059
    assert p['enable_interactive_mode'] is False


def test_ekf_and_map_configs_parse():
    assert _load('ekf.yaml')['ekf_filter_node']['ros__parameters']['world_frame'] == 'odom'
    assert 'Visualization Manager' in _load('map_view.rviz')
