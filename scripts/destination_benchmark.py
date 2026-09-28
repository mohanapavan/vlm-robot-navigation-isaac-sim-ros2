"""Destination benchmark on the REAL stack: N easy, N medium and N hard destinations, typed to Qwen as sentences.

Difficulty is the length of the path Nav2's planner finds from wherever the robot is when the destination is chosen:
    easy 3-12 m   |   medium 12-30 m   |   hard > 30 m (preferring groups and spots tight against a wall)
Each tier prefers classes not yet used in it. A destination PASSES when the model's reply is understood, Nav2 reports
SUCCEEDED, the robot ends within 0.6 m of the goal point it was sent to, and it faces the object. Because the scene
graph can contain look-alike detections, each destination is also checked against the simulator's ground truth:
"real" says whether the entry the robot drove to is next to a real object of that class.

Needs Isaac Sim playing, SLAM (saved map) + Nav2 up (scripts/pipeline.sh start) and the Qwen environment:

    source /opt/ros/humble/setup.bash && source ~/ws/install/setup.bash
    ~/qwen_env/bin/python scripts/destination_benchmark.py [--per-tier 5] [--seed 1]
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import rclpy
from nav2_msgs.action import ComputePathToPose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_command_test import run_command  # noqa: E402
from vlm_nav.evaluation import distance_to_footprint, load_ground_truth  # noqa: E402
from vlm_nav.geometry import normalize_angle, yaw_facing  # noqa: E402
from vlm_nav.robot_brain import QwenChat, RobotBrain, load_graph  # noqa: E402

TIERS = [('easy', 3.0, 12.0), ('medium', 12.0, 30.0), ('hard', 30.0, math.inf)]
TIMEOUT = {'easy': 420.0, 'medium': 700.0, 'hard': 1100.0}      # wall seconds per goal (the simulator is slower than real time)
TEMPLATES = ['go to {ref}', 'take me to {ref}', 'navigate to {ref}', 'head to {ref}', 'go to {ref}']
GT_PATH = Path(__file__).resolve().parents[1] / 'evaluation' / 'hospital_ground_truth.json'


class Odometer(threading.Thread):
    """Distance actually driven, from the map -> base_link pose sampled at 5 Hz."""

    def __init__(self, brain):
        super().__init__(daemon=True)
        self.brain, self.distance, self._stop_flag, self._last = brain, 0.0, threading.Event(), None

    def run(self):
        while not self._stop_flag.is_set():
            pose = self.brain.get_pose()
            if pose:
                if self._last is not None:
                    step = math.hypot(pose[0] - self._last[0], pose[1] - self._last[1])
                    self.distance += step if step < 1.0 else 0.0            # ignore a TF glitch
                self._last = pose
            time.sleep(0.2)

    def stop(self):
        self._stop_flag.set()


def plan_length(brain, x, y):
    """Length (m) of Nav2's path from the robot to (x, y), or None when there is none."""
    if not brain.plan_client.wait_for_server(timeout_sec=2.0):
        return None
    goal = ComputePathToPose.Goal()
    goal.goal.header.frame_id = 'map'
    goal.goal.pose.position.x, goal.goal.pose.position.y, goal.goal.pose.orientation.w = float(x), float(y), 1.0
    goal.use_start = False
    sent = brain.plan_client.send_goal_async(goal)
    if not brain._wait(sent, 5.0) or not sent.result().accepted:
        return None
    result = sent.result().get_result_async()
    if not brain._wait(result, 15.0):
        return None
    poses = result.result().result.path.poses
    if len(poses) < 1:
        return None
    return sum(math.hypot(b.pose.position.x - a.pose.position.x, b.pose.position.y - a.pose.position.y)
               for a, b in zip(poses, poses[1:]))


def candidates(brain, graph, visited):
    """Every unvisited object with a reachable approach point, its planned path length and how tight the spot is."""
    x, y, _ = brain.get_pose()
    found = []
    for o in graph.objects:
        if o.id in visited:
            continue
        choice = brain.choose_approach(o.id, (x, y))
        if choice in (None, 'unknown'):
            continue
        _, gx, gy = choice
        length = plan_length(brain, gx, gy)
        if length is None:
            continue
        found.append({'obj': o, 'approach': (gx, gy), 'length': length,
                      'tight': brain.grid.clearance(gx, gy) < 0.7 if brain.grid else False,
                      'nearest_of_label': graph.resolve(o.label, (x, y)) is o})
    return found


def pick(cands, tier, lo, hi, used_labels, rng):
    pool = [c for c in cands if lo <= c['length'] < hi]
    if not pool:                                  # nothing in the band from here: take what is closest to it
        target = lo if math.isinf(hi) else (lo + hi) / 2
        pool = sorted(cands, key=lambda c: abs(c['length'] - target))[:4]
    fresh = [c for c in pool if c['obj'].label not in used_labels] or pool
    if tier == 'hard':
        fresh = [c for c in fresh if c['obj'].kind == 'group' or c['tight']] or fresh
    return rng.choice(sorted(fresh, key=lambda c: c['obj'].id))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene-graph', default=os.path.expanduser('~/scene_graph/scene_graph.json'))
    p.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--per-tier', type=int, default=5)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--output', default=os.path.expanduser('~/pipeline_logs/destination_benchmark.json'))
    args = p.parse_args()
    rng = random.Random(args.seed)

    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    graph = load_graph(args.scene_graph, True, os.path.expanduser('~/my_map.yaml'))
    gt = load_ground_truth(GT_PATH)
    brain = RobotBrain(graph, standoff=0.8, server_timeout=10.0)
    threading.Thread(target=rclpy.spin, args=(brain,), daemon=True).start()
    while brain.get_pose() is None or brain.grid is None:
        time.sleep(0.2)
    print(f'{len(graph)} objects; robot at {tuple(round(v, 2) for v in brain.get_pose())}; loading Qwen...', flush=True)
    llm = QwenChat(args.model)
    odo = Odometer(brain)
    odo.start()

    rows, visited, n = [], set(), 0
    for tier, lo, hi in TIERS:
        used = set()
        for k in range(args.per_tier):
            n += 1
            cands = candidates(brain, graph, visited)
            c = pick(cands, tier, lo, hi, used, rng)
            o = c['obj']
            used.add(o.label)
            visited.add(o.id)
            ref = f'the {o.label}' if c['nearest_of_label'] and rng.random() < 0.5 else o.id
            command = TEMPLATES[k % len(TEMPLATES)].format(ref=ref)
            start = brain.get_pose()
            d0, sim0, wall0 = odo.distance, brain.get_clock().now().nanoseconds * 1e-9, time.time()
            print(f'\n[{n:2d}] {tier.upper():6s} "{command}"  -> {o.id} ({o.kind}), planned path {c["length"]:.1f} m', flush=True)
            reply, out, sent, res = run_command(brain, llm, command, TIMEOUT[tier])
            end = brain.get_pose()
            goal = brain.last_goal if sent else None
            row = {'n': n, 'tier': tier, 'command': command, 'target': o.id, 'label': o.label, 'kind': o.kind,
                   'reply': reply, 'planned_m': round(c['length'], 1), 'result': res or 'NO GOAL',
                   'wall_s': round(time.time() - wall0), 'sim_s': round(brain.get_clock().now().nanoseconds * 1e-9 - sim0),
                   'driven_m': round(odo.distance - d0, 1), 'start': [round(start[0], 1), round(start[1], 1)],
                   'notes': ' | '.join(x.strip() for x in out if '[NAV]' in x)[:200]}
            reach = o.radius + 1.5 if o.kind == 'group' else 1.5
            same = [g for g in gt if g['class'] == o.label]
            row['real'] = any(distance_to_footprint(o.x, o.y, g) <= reach for g in same) if same else None
            row['final_to_real_m'] = round(
                min((distance_to_footprint(end[0], end[1], g) for g in same), default=float('nan')), 1)
            reached_target = sent and brain.last_goal is not None
            if reached_target:
                target_obj = graph.resolve(o.id)
                row['goal_err_m'] = round(math.hypot(end[0] - goal[0], end[1] - goal[1]), 2)
                row['facing_err_deg'] = round(math.degrees(abs(normalize_angle(
                    yaw_facing((end[0], end[1]), target_obj.xy) - end[2]))))
                row['went_to'] = ' '.join(x for x in row['notes'].split("'")[1:2])
                row['pass'] = (res == 'SUCCEEDED' and row['goal_err_m'] <= 0.6 and row['facing_err_deg'] <= 40
                               and row['went_to'] == o.id)
            else:
                row['pass'] = False
            rows.append(row)
            print(f'     reply "{reply}" -> {row["result"]}; driven {row["driven_m"]} m in {row["sim_s"]} sim-s '
                  f'({row["wall_s"]} wall-s); goal error {row.get("goal_err_m")} m, facing error '
                  f'{row.get("facing_err_deg")} deg; real object: {row["real"]}  => '
                  f'{"PASS" if row["pass"] else "FAIL"}', flush=True)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(rows, indent=1))

    odo.stop()
    print('\n' + '=' * 110)
    print(f'{"#":>2} {"tier":6s} {"command":34s} {"result":10s} {"plan m":>6s} {"drove m":>7s} {"sim s":>5s} '
          f'{"err m":>5s} {"face°":>5s} {"real":>5s}  verdict')
    for r in rows:
        print(f'{r["n"]:2d} {r["tier"]:6s} {r["command"][:34]:34s} {r["result"]:10s} {r["planned_m"]:6.1f} {r["driven_m"]:7.1f} '
              f'{r["sim_s"]:5d} {str(r.get("goal_err_m", "-")):>5s} {str(r.get("facing_err_deg", "-")):>5s} '
              f'{str(r["real"]):>5s}  {"PASS" if r["pass"] else "FAIL"}')
    print('=' * 110)
    for tier, _, _ in TIERS:
        sel = [r for r in rows if r['tier'] == tier]
        print(f'{tier:7s} {sum(r["pass"] for r in sel)}/{len(sel)} passed; destination is a real object for '
              f'{sum(bool(r["real"]) for r in sel)}/{len(sel)}')
    total = sum(r['pass'] for r in rows)
    print(f'TOTAL   {total}/{len(rows)} passed; real destinations {sum(bool(r["real"]) for r in rows)}/{len(rows)}')
    brain.cancel_current_goal_if_active()
    rclpy.shutdown()
    time.sleep(0.5)
    brain.destroy_node()
    return 0 if total == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
