"""End-to-end smoke test of the brain -> Nav2 path on the *running* stack (no LLM needed).

Prerequisites: Isaac Sim playing, `ros2 launch vlm_nav slam.launch.py` and `ros2 launch vlm_nav navigation.launch.py`
running, and the robot standing in open, already-mapped floor.

It drives the real RobotBrain / chat_loop code with scripted model replies, and asks Nav2's own planner which
targets are reachable so it never sends a goal into a wall:

    python3 scripts/nav_smoke_test.py

Checks (each prints PASS/FAIL, exit code 0 only if all pass):
  relative         RELATIVE:(dx,dy) moves dx forward / dy left of the robot's *current heading*
  goal             GOAL:<name> stops ~standoff metres short of the object and faces it
  stop             STOP mid-drive stops the robot
  goal_after_stop  a new goal still works after a cancel
"""
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_nav.geometry import normalize_angle, rotate_relative_offset  # noqa: E402
from vlm_nav.robot_brain import RobotBrain, chat_loop  # noqa: E402
from vlm_nav.scene_graph import SceneGraph, SceneObject  # noqa: E402

GOAL_TIMEOUT = 300.0        # wall seconds; the simulator can run well below real time


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    graph = SceneGraph([])
    brain = RobotBrain(graph, standoff=0.8, server_timeout=10.0)
    threading.Thread(target=rclpy.spin, args=(brain,), daemon=True).start()
    while brain.get_pose() is None:
        time.sleep(0.2)
    planner = ActionClient(brain, ComputePathToPose, 'compute_path_to_pose')
    if not planner.wait_for_server(timeout_sec=10):
        sys.exit('Nav2 planner not available: is navigation.launch.py running?')

    def reachable(gx, gy):
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = 'map'
        goal.goal.pose.position.x, goal.goal.pose.position.y, goal.goal.pose.orientation.w = gx, gy, 1.0
        goal.use_start = False
        fut = planner.send_goal_async(goal)
        t = time.time()
        while not fut.done() and time.time() - t < 10:
            time.sleep(0.05)
        handle = fut.result()
        if handle is None or not handle.accepted:
            return False
        res = handle.get_result_async()
        t = time.time()
        while not res.done() and time.time() - t < 15:
            time.sleep(0.05)
        return res.done() and len(res.result().result.path.poses) > 5

    def find_reachable(x, y, distances):
        for d in distances:
            for ang in np.linspace(-math.pi, math.pi, 24, endpoint=False):
                gx, gy = x + d * math.cos(ang), y + d * math.sin(ang)
                if reachable(gx, gy):
                    return gx, gy, ang
        return None

    def say(reply, user_line):
        """Feed one user line through chat_loop with a scripted model reply; return once the goal is sent."""
        lines = iter([user_line, 'exit'])
        out = []
        chat_loop(brain, lambda sp, ut, img: reply, input_fn=lambda _: next(lines), output_fn=out.append)
        for o in out:
            print('   ', o.replace('\n', ' | '))

    def wait_result():
        return brain.last_result if brain.result_event.wait(GOAL_TIMEOUT) else 'TIMEOUT'

    results = {}
    x0, y0, yaw0 = brain.get_pose()
    print(f'start pose in map: ({x0:.2f}, {y0:.2f}) yaw {math.degrees(yaw0):.0f} deg')

    print('\n[relative] heading-aware relative move')
    offset = next(((dx, dy) for dx, dy in [(1.5, 0), (0, 1.5), (0, -1.5), (-1.5, 0), (1, 0), (0, 1), (0, -1), (-1, 0)]
                   if reachable(*rotate_relative_offset(x0, y0, yaw0, dx, dy))), None)
    print('   planner-verified reachable offset (forward, left):', offset)
    if offset:
        say(f'RELATIVE:({offset[0]},{offset[1]})\nMoving.', 'move')
        res = wait_result()
        x1, y1, yaw1 = brain.get_pose()
        fwd = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
        left = -(x1 - x0) * math.sin(yaw0) + (y1 - y0) * math.cos(yaw0)
        print(f'   {res}: moved forward {fwd:+.2f} left {left:+.2f} (asked {offset[0]:+.2f}, {offset[1]:+.2f})')
        results['relative'] = res == 'SUCCEEDED' and math.hypot(fwd - offset[0], left - offset[1]) < 0.45
    else:
        results['relative'] = False

    print('\n[goal] named object; the model reply mentions "stop" in prose')
    x1, y1, _ = brain.get_pose()
    spot = find_reachable(x1, y1, (2.5, 3.0, 2.0, 3.5, 1.5, 4.0))
    print('   planner-verified stand-off cell:', spot and tuple(round(v, 2) for v in spot))
    if spot:
        gx, gy, ang = spot
        ox, oy = gx + 0.8 * math.cos(ang), gy + 0.8 * math.sin(ang)
        graph.objects.append(SceneObject('forklift_1', 'forklift', ox, oy, 0.5, 5, 0.7))
        say("GOAL:forklift\nI'll go there and stop in front of it.", 'go to the forklift')
        res = wait_result()
        px, py, pyaw = brain.get_pose()
        dist = math.hypot(ox - px, oy - py)
        face = abs(normalize_angle(math.atan2(oy - py, ox - px) - pyaw))
        print(f'   {res}: {dist:.2f} m from the object (standoff 0.8 + Nav2 tolerance), '
              f'facing error {math.degrees(face):.0f} deg')
        results['goal'] = res == 'SUCCEEDED' and abs(dist - 0.8) < 0.45 and face < 0.5
    else:
        results['goal'] = False

    print('\n[stop] STOP mid-drive, then a fresh goal')
    x2, y2, _ = brain.get_pose()
    far = find_reachable(x2, y2, (3.5, 3.0, 2.5, 2.0))
    print('   far reachable goal:', far and tuple(round(v, 2) for v in far[:2]))
    if far:
        graph.objects.append(SceneObject('shelf_1', 'shelf', far[0], far[1], 0.5, 5, 0.7))
        say('GOAL:shelf_1\nOn my way.', 'go to the shelf')
        time.sleep(12)                                       # let it get going
        say('STOP\nStopping.', 'stop')
        pa = brain.get_pose()
        time.sleep(5)
        pb = brain.get_pose()
        moved, drift = math.hypot(pa[0] - x2, pa[1] - y2), math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        print(f'   moved {moved:.2f} m before STOP; drift after STOP {drift:.3f} m')
        results['stop'] = moved > 0.05 and drift < 0.1
        time.sleep(3)
        say('GOAL:shelf_1\nGoing again.', 'go to the shelf again')
        res = wait_result()
        print(f'   fresh goal after STOP -> {res}')
        results['goal_after_stop'] = res == 'SUCCEEDED'
    else:
        results['stop'] = results['goal_after_stop'] = False

    print('\nRESULT:', ' | '.join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in results.items()))
    rclpy.shutdown()
    time.sleep(0.5)
    brain.destroy_node()
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
