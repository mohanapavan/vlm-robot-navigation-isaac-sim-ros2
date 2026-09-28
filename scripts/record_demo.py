"""Record a demo video + GIF of the real stack: commands typed to Qwen, the robot's camera and its position on the saved map.

Needs Isaac Sim playing, SLAM (saved map) + Nav2 up (scripts/pipeline.sh start) and the Qwen environment:

    source /opt/ros/humble/setup.bash && source ~/ws/install/setup.bash
    ~/qwen_env/bin/python scripts/record_demo.py --out docs/media/hospital_demo \\
        --command "what do you see?" --command "go to vending_machine_3" --command "what do you see?"

Every command goes through the real chat loop (Qwen2.5-VL -> parser -> Nav2). A frame is captured every `--frame-sim-s` seconds
of *simulated* time (default 0.5) from the front camera and the map -> base_link pose, and played back at `--fps` (default 10),
so the video runs (frame-sim-s * fps) times faster than the simulation: 5x by default. Writes <out>.mp4 and <out>.gif.
"""
import argparse
import math
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_command_test import run_command  # noqa: E402
from plot_scene_graph import CLASS_BGR, Canvas  # noqa: E402
from vlm_nav.robot_brain import QwenChat, RobotBrain, load_graph  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANEL_W, PANEL_H, BAR_H = 480, 300, 74


class Frames:
    """Composes one video frame: camera | map with trail, under a bar with the command, the reply and the status."""

    def __init__(self, graph, map_yaml):
        canvas = Canvas(map_yaml, scale=0.6, margin=1.0, include=[(o.x, o.y) for o in graph.objects])
        base = canvas.img.copy()
        for o in graph.objects:
            x, y = canvas.px(o.x, o.y)
            color = CLASS_BGR.get(o.label, (0, 0, 0))
            if o.kind == 'group':
                cv2.circle(base, (x, y), max(canvas.metres(o.radius), 5), color, 1, cv2.LINE_AA)
            cv2.circle(base, (x, y), 3, color, -1, cv2.LINE_AA)
        scale = min(PANEL_W / base.shape[1], PANEL_H / base.shape[0])
        self.k = scale * 1.0
        base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        panel = np.full((PANEL_H, PANEL_W, 3), 236, np.uint8)
        self.dx, self.dy = (PANEL_W - base.shape[1]) // 2, (PANEL_H - base.shape[0]) // 2
        panel[self.dy:self.dy + base.shape[0], self.dx:self.dx + base.shape[1]] = base
        self.base, self.canvas, self.trail = panel, canvas, []

    def to_px(self, x, y):
        px, py = self.canvas.px(x, y)
        return int(px * self.k) + self.dx, int(py * self.k) + self.dy

    def compose(self, camera_rgb, pose, goal, command, reply, status):
        panel = self.base.copy()
        here = self.to_px(pose[0], pose[1])
        self.trail.append(here)
        if len(self.trail) > 1:
            cv2.polylines(panel, [np.array(self.trail, np.int32)], False, (200, 90, 0), 2, cv2.LINE_AA)
        if goal is not None:
            cv2.drawMarker(panel, self.to_px(goal[0], goal[1]), (0, 0, 220), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
        tip = (int(here[0] + 11 * math.cos(-pose[2])), int(here[1] + 11 * math.sin(-pose[2])))
        cv2.circle(panel, here, 6, (0, 150, 0), -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, here, tip, (0, 90, 0), 2, cv2.LINE_AA, tipLength=0.5)
        cam = cv2.resize(camera_rgb[:, :, ::-1], (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA) \
            if camera_rgb is not None else np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
        bar = np.full((BAR_H, 2 * PANEL_W, 3), 255, np.uint8)
        lines = [f'You: {command}' if command else 'You: ...']
        lines += textwrap.wrap(f'Robot: {reply}', 105)[:2] if reply else ['Robot: ...']
        lines.append(status)
        for i, line in enumerate(lines[:4]):
            cv2.putText(bar, line, (8, 16 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20) if i < 3 else (0, 110, 0), 1,
                        cv2.LINE_AA)
        return np.vstack([bar, np.hstack([cam, panel])])


def status_text(brain, goal):
    if goal is None:
        return 'idle'
    x, y, _ = brain.get_pose() or (0, 0, 0)
    return (f'driving, {math.hypot(goal[0] - x, goal[1] - y):.1f} m from the goal point '
            '  (the video plays 5x faster than the simulation)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(ROOT / 'docs' / 'media' / 'hospital_demo'))
    p.add_argument('--command', action='append', required=True, help='one typed command; repeat for several')
    p.add_argument('--scene-graph', default=str(ROOT / 'saved_state' / 'hospital' / 'scene_graph.consolidated.json'))
    p.add_argument('--map', default=str(ROOT / 'saved_state' / 'hospital' / 'my_map.yaml'))
    p.add_argument('--frame-sim-s', type=float, default=0.5)
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--gif-width', type=int, default=800)
    p.add_argument('--hold', type=int, default=20, help='frames to keep each finished command on screen')
    args = p.parse_args()

    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    graph = load_graph(args.scene_graph, True, args.map, output_fn=lambda *a: None)
    brain = RobotBrain(graph, standoff=0.8, server_timeout=10.0)
    threading.Thread(target=rclpy.spin, args=(brain,), daemon=True).start()
    while brain.get_pose() is None or brain.grid is None or brain.get_latest_image(640) is None:
        time.sleep(0.2)
    frames = Frames(graph, args.map)
    llm = QwenChat()

    state = {'command': '', 'reply': '', 'goal': None}
    captured, stop, paused = [], threading.Event(), threading.Event()

    def snapshot():
        img = brain.get_latest_image(640)
        return frames.compose(np.array(img) if img is not None else None, brain.get_pose(), state['goal'], state['command'],
                              state['reply'], status_text(brain, state['goal']))

    def capture():
        last = None
        while not stop.is_set():
            now = brain.get_clock().now().nanoseconds * 1e-9
            if not paused.is_set() and brain.get_pose() and (last is None or now - last >= args.frame_sim_s):
                last = now
                captured.append(snapshot())
            time.sleep(0.02)

    thread = threading.Thread(target=capture, daemon=True)
    thread.start()
    for command in args.command:
        state.update(command=command, reply='(thinking...)', goal=None)
        n_goals = brain.goals_sent
        result = {}

        def ask(command=command, result=result):
            result['out'] = run_command(brain, llm, command, 900.0)
        worker = threading.Thread(target=ask)
        worker.start()
        while worker.is_alive():
            if brain.goals_sent > n_goals and state['goal'] is None and brain.last_goal is not None:
                state['goal'] = brain.last_goal
            if state['reply'] == '(thinking...)' and brain.goals_sent > n_goals:
                state['reply'] = 'on my way'
            time.sleep(0.1)
        reply, out, sent, res = result['out']
        state.update(reply=reply or '(no reply)', goal=None)
        note = ' | '.join(o.strip() for o in out if '[NAV]' in o)
        end_status = f'{res}' if sent else (note[:100] or 'answered in words, robot did not move')
        state['reply'] = reply or '(no reply)'
        state['command'] = command + f'   ->  {end_status}'
        paused.set()                                  # keep the finished command (answer / arrival) on screen
        time.sleep(0.3)
        captured.extend([snapshot()] * args.hold)
        paused.clear()
    stop.set()
    thread.join()
    brain.cancel_current_goal_if_active()
    rclpy.shutdown()

    h, w = captured[0].shape[:2]
    mp4_ok = False
    for fourcc in ('avc1', 'mp4v'):
        writer = cv2.VideoWriter(args.out + '.mp4', cv2.VideoWriter_fourcc(*fourcc), args.fps, (w, h))
        if writer.isOpened():
            for f in captured:
                writer.write(f)
            writer.release()
            mp4_ok = True
            print(f'wrote {args.out}.mp4 ({fourcc}, {len(captured)} frames, {w}x{h})')
            break
    gw = args.gif_width
    small = [Image.fromarray(cv2.cvtColor(cv2.resize(f, (gw, int(h * gw / w)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
             .convert('P', palette=Image.ADAPTIVE, colors=48) for f in captured[::2]]
    small[0].save(args.out + '.gif', save_all=True, append_images=small[1:], duration=int(2000 / args.fps), loop=0, optimize=True)
    print(f'wrote {args.out}.gif ({len(small)} frames, {gw} px wide, {os.path.getsize(args.out + ".gif") / 1e6:.1f} MB)')
    cv2.imwrite(args.out + '_still.png', captured[len(captured) // 2])
    return 0 if mp4_ok else 1


if __name__ == '__main__':
    sys.exit(main())
