"""Does the live lidar scan line up with the saved map? (Checks the identity map -> odom used by `mode:=static`.)

Needs Isaac Sim playing and `slam_lidar.launch.py` running. Transforms the scan into the map frame with the live TF and
reports how many scan endpoints fall on (or right next to) occupied cells of the saved map:

    python3 scripts/check_map_alignment.py [--map ~/my_map.yaml] [--scans 5]

Exit code 0 when at least --min-hit of the endpoints match; otherwise the pose the robot believes it has is wrong
(the simulator was not reset with Stop -> Play, or the map belongs to another start pose).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_nav.geometry import quat_to_yaw  # noqa: E402
from vlm_nav.map_grid import MapGrid  # noqa: E402


class ScanChecker(Node):
    def __init__(self):
        super().__init__('check_map_alignment')
        self.scans = []
        self.tf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf, self)
        self.create_subscription(LaserScan, '/scan', self.scans.append, qos_profile_sensor_data)


def scan_points_in_map(node, scan, map_frame='map'):
    """Scan endpoints (N, 2) in the map frame, or None if the TF is not available."""
    try:
        t = node.tf.lookup_transform(map_frame, scan.header.frame_id, Time())
    except tf2_ros.TransformException:
        return None
    yaw = quat_to_yaw(*(getattr(t.transform.rotation, k) for k in 'xyzw'))
    r = np.array(scan.ranges, dtype=float)
    ang = scan.angle_min + np.arange(len(r)) * scan.angle_increment
    ok = np.isfinite(r) & (r > max(scan.range_min, 0.5)) & (r < min(scan.range_max, 25.0))
    x, y = r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([t.transform.translation.x + c * x - s * y, t.transform.translation.y + s * x + c * y], axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', default=os.path.expanduser('~/my_map.yaml'))
    p.add_argument('--scans', type=int, default=5, help='scans to average')
    p.add_argument('--tol', type=float, default=0.15, help='metres from an occupied cell that counts as a match')
    p.add_argument('--min-hit', type=float, default=0.6)
    p.add_argument('--timeout', type=float, default=30.0)
    args = p.parse_args()

    grid = MapGrid.from_yaml(args.map)
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ScanChecker()
    fractions, end = [], time.time() + args.timeout
    seen = 0
    while len(fractions) < args.scans and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
        while seen < len(node.scans) and len(fractions) < args.scans:
            pts = scan_points_in_map(node, node.scans[seen])
            seen += 1
            if pts is not None and len(pts):
                fractions.append(np.mean([grid.clearance(x, y) <= args.tol for x, y in pts]))
                time.sleep(0.3)
    node.destroy_node()
    rclpy.shutdown()
    if not fractions:
        print('No scan could be placed in the map frame: is /scan published and is the map -> base_link TF up?')
        return 2
    hit = float(np.mean(fractions))
    print(f'{len(fractions)} scans: {hit:.0%} of the lidar endpoints lie within {args.tol} m of an occupied cell of {args.map}')
    print('ALIGNED' if hit >= args.min_hit else 'MISALIGNED: the robot is not where the saved map says it is')
    return 0 if hit >= args.min_hit else 1


if __name__ == '__main__':
    sys.exit(main())
