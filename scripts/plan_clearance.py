"""How far from mapped obstacles does Nav2's planned path stay? (The probe behind the inflation change, CHANGELOG 36.)

Asks the running planner for a path between two points *without moving the robot* and measures its clearance on the saved map:

    python3 scripts/plan_clearance.py START_X START_Y GOAL_X GOAL_Y [--map ~/my_map.yaml]

Compare before / after changing `inflation_radius` (in config/nav2_params_lidar.yaml, or live with
`ros2 param set /global_costmap/global_costmap inflation_layer.inflation_radius 0.9`). For the plan from the chair_1 stand-off
(-14.3, 2.68) to the vending machine (-31.7, 3.3): inflation 0.5 m -> 17.5 m long, closest approach 0.51 m, 21 % of the path
under 0.7 m; inflation 0.9 m -> 17.8 m, closest 0.77 m, 0 % under 0.7 m.
"""
import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import rclpy
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_nav.map_grid import MapGrid  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('start_x', type=float)
    p.add_argument('start_y', type=float)
    p.add_argument('goal_x', type=float)
    p.add_argument('goal_y', type=float)
    p.add_argument('--map', default=os.path.expanduser('~/my_map.yaml'))
    args = p.parse_args()
    grid = MapGrid.from_yaml(os.path.expanduser(args.map))

    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = Node('plan_clearance')
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    client = ActionClient(node, ComputePathToPose, '/compute_path_to_pose')
    if not client.wait_for_server(timeout_sec=10):
        sys.exit('Nav2 planner not available: is scripts/pipeline.sh start running?')
    goal = ComputePathToPose.Goal()
    goal.use_start = True
    goal.start.header.frame_id = goal.goal.header.frame_id = 'map'
    goal.start.pose.position.x, goal.start.pose.position.y, goal.start.pose.orientation.w = args.start_x, args.start_y, 1.0
    goal.goal.pose.position.x, goal.goal.pose.position.y, goal.goal.pose.orientation.w = args.goal_x, args.goal_y, 1.0
    sent = client.send_goal_async(goal)
    while not sent.done():
        time.sleep(0.05)
    if not sent.result().accepted:
        sys.exit('the planner rejected the request')
    result = sent.result().get_result_async()
    while not result.done():
        time.sleep(0.05)
    pts = [(q.pose.position.x, q.pose.position.y) for q in result.result().result.path.poses]
    if len(pts) < 2:
        sys.exit('no path')
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    clearance = [grid.clearance(*q) for q in pts]
    early = [c for c, d in zip(clearance, cum) if 0.6 < d < 6.0]
    print(f'path {cum[-1]:.1f} m; clearance from mapped obstacles: closest {min(clearance):.2f} m, '
          f'closest in 0.6-6 m out of the start {min(early) if early else float("nan"):.2f} m; '
          f'under 0.7 m: {sum(c < 0.7 for c in clearance) / len(clearance):.0%} of the path, '
          f'under 0.55 m: {sum(c < 0.55 for c in clearance) / len(clearance):.0%}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
