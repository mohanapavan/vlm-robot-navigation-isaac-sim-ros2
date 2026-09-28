"""Run natural-language commands through the REAL stack and check each against explicit pass criteria.

Needs: Isaac Sim playing the scene, SLAM (saved map) + Nav2 up (scripts/pipeline.sh start), and the Qwen environment:

    source /opt/ros/humble/setup.bash && source ~/ws/install/setup.bash
    ~/qwen_env/bin/python scripts/live_command_test.py [--scene-graph ~/scene_graph/scene_graph.json]

Every command is typed into the real chat loop (Qwen2.5-VL reads the prompt, the brain parses the reply and drives Nav2).
Exit code 0 only if every command passes.
"""
import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_nav.geometry import normalize_angle, yaw_facing  # noqa: E402
from vlm_nav.robot_brain import QwenChat, RobotBrain, chat_loop, load_graph  # noqa: E402

GOAL_TIMEOUT = 420.0        # wall seconds per goal: the simulator runs at a fraction of real time


def run_command(brain, llm, text, timeout=GOAL_TIMEOUT):
    """Type one line into the chat loop. Returns (model reply, brain output lines, goal sent or None, result)."""
    out = []
    lines = iter([text, 'exit'])
    before = brain.goals_sent
    brain.result_event.clear()
    chat_loop(brain, llm, input_fn=lambda _: next(lines), output_fn=out.append, wait_for_result=True,
              result_timeout=timeout)
    reply = next((o.strip().replace('Robot: ', '', 1) for o in out if o.strip().startswith('Robot:')), '')
    sent = brain.last_goal if brain.goals_sent > before else None
    return reply, out, sent, (brain.last_result if sent else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-graph', default=os.path.expanduser('~/scene_graph/scene_graph.json'))
    p.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
    args = p.parse_args()

    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    graph = load_graph(args.scene_graph, True, os.path.expanduser('~/my_map.yaml'))
    brain = RobotBrain(graph, standoff=0.8, server_timeout=10.0)
    threading.Thread(target=rclpy.spin, args=(brain,), daemon=True).start()
    while brain.get_pose() is None:
        time.sleep(0.2)
    while brain.grid is None:
        time.sleep(0.2)
    print(f'{len(graph)} objects known; robot at {tuple(round(v, 2) for v in brain.get_pose())}; loading Qwen...', flush=True)
    llm = QwenChat(args.model)

    checks = []       # (command, passed, detail)

    # 1 ---- relative move: heading-aware, ends about where asked
    x0, y0, yaw0 = brain.get_pose()
    reply, out, sent, res = run_command(brain, llm, 'go forward 2 meters')
    x1, y1, _ = brain.get_pose()
    fwd = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
    lat = -(x1 - x0) * math.sin(yaw0) + (y1 - y0) * math.cos(yaw0)
    ok = sent is not None and res == 'SUCCEEDED' and abs(fwd - 2.0) < 0.5 and abs(lat) < 0.4
    checks.append(('go forward 2 meters', ok, f'reply "{reply}" -> {res}; moved {fwd:+.2f} m forward, {lat:+.2f} m sideways'))

    # 2 ---- named goal: reaches the stand-off point and faces the object
    x0, y0, _ = brain.get_pose()
    reply, out, sent, res = run_command(brain, llm, 'go to the wheelchair')
    obj = graph.resolve('wheelchair', (x0, y0))
    if sent and obj:
        x1, y1, yaw1 = brain.get_pose()
        dist = math.hypot(obj.x - x1, obj.y - y1)
        face = abs(normalize_angle(yaw_facing((x1, y1), obj.xy) - yaw1))
        ok = res == 'SUCCEEDED' and abs(dist - 0.8) < 0.5 and face < 0.6
        detail = (f'reply "{reply}" -> {res}; {dist:.2f} m from {obj.id} (stand-off 0.8), '
                  f'facing error {math.degrees(face):.0f} deg')
    else:
        ok, detail = False, f'reply "{reply}"; nothing was sent to Nav2: ' + ' | '.join(o.strip() for o in out if '[NAV]' in o)
    checks.append(('go to the wheelchair', ok, detail))

    # 3 ---- vision question: answers in words, never moves
    pa = brain.get_pose()
    reply, out, sent, res = run_command(brain, llm, 'what do you see?')
    pb = brain.get_pose()
    moved = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
    is_command = reply.upper().startswith(('GOAL', 'MOVE', 'RELATIVE', 'STOP'))
    ok = sent is None and moved < 0.05 and len(reply) > 15 and not is_command
    goals = 0 if sent is None else 1
    checks.append(('what do you see?', ok, f'reply "{reply[:160]}"; robot moved {moved:.3f} m, goals sent: {goals}'))

    # 4 ---- unknown object: refuses, sends nothing
    pa = brain.get_pose()
    reply, out, sent, res = run_command(brain, llm, 'go to the spaceship')
    pb = brain.get_pose()
    moved = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
    ok = sent is None and moved < 0.05
    goals = 0 if sent is None else 1
    checks.append(('go to the spaceship', ok, f'reply "{reply[:120]}"; goals sent: {goals}, moved {moved:.3f} m'))

    # 5 ---- a group entry: goes to its edge, not into the pile
    here = brain.get_pose()
    far_groups = [o for o in graph.objects if o.kind == 'group'
                  and math.hypot(o.x - here[0], o.y - here[1]) > 0.8 + o.radius + 1.5]      # not already standing at it
    group = min(far_groups, key=lambda o: math.hypot(o.x - here[0], o.y - here[1]), default=None)
    if group is None:
        checks.append(('go to <group>', False, 'the scene graph has no group entry'))
    else:
        reply, out, sent, res = run_command(brain, llm, f'go to {group.id}')
        if sent:
            x1, y1, yaw1 = brain.get_pose()
            dist = math.hypot(group.x - x1, group.y - y1)
            expect = 0.8 + group.radius
            ok = res == 'SUCCEEDED' and abs(dist - expect) < 0.7
            detail = (f'reply "{reply}" -> {res}; {dist:.2f} m from the centre of {group.id} '
                      f'(group of {group.members}, {group.size:.1f} m wide; expected about {expect:.1f} m)')
        else:
            ok, detail = False, f'reply "{reply}"; nothing was sent: ' + ' | '.join(o.strip() for o in out if '[NAV]' in o)
        checks.append((f'go to {group.id}', ok, detail))

    print('\n' + '=' * 78)
    for i, (cmd, ok, detail) in enumerate(checks, 1):
        print(f'{i}. {"PASS" if ok else "FAIL"}  "{cmd}"\n     {detail}')
    print('=' * 78)
    passed = sum(ok for _, ok, _ in checks)
    print(f'{passed}/{len(checks)} commands passed')
    brain.cancel_current_goal_if_active()
    rclpy.shutdown()
    time.sleep(0.5)
    brain.destroy_node()
    return 0 if passed == len(checks) else 1


if __name__ == '__main__':
    sys.exit(main())
