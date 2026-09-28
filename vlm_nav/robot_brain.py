"""Qwen2.5-VL natural-language robot brain: chat -> GOAL / RELATIVE / STOP -> Nav2.

Run (Nav2 + the SLAM pipeline must already be up, see docs/commands.md):
    ros2 run vlm_nav robot_brain --scene-graph ~/scene_graph/scene_graph.json --ros-args -p use_sim_time:=true
"""
import argparse
import math
import os
import sys
import threading
import time

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from PIL import Image as PILImage
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image as RosImage

from .command_parser import (build_question_prompt, build_system_prompt, first_command_line, is_question,
                             mentioned_objects, needs_vision, parse_response, reconcile_goal)
from .geometry import quat_to_yaw, ring_points, rotate_relative_offset, standoff_point, yaw_facing, yaw_to_quat
from .image_utils import image_msg_to_rgb
from .map_grid import FREE, MapGrid
from .scene_graph import SceneGraph


def load_graph(path, consolidate_raw=True, map_yaml=None, output_fn=print):
    """Load a scene graph; a raw one (one entry per detection cluster) is consolidated first so the model is told about
    a few trustworthy objects, not hundreds of fragments (see consolidate.py)."""
    graph = SceneGraph.load(path)
    if graph.dropped_on_load:
        output_fn(f'Warning: skipped {graph.dropped_on_load} malformed object(s) in {path}')
    if graph.consolidated or not consolidate_raw:
        return graph
    from .consolidate import consolidate
    grid = None
    if map_yaml and os.path.isfile(os.path.expanduser(map_yaml)):
        grid = MapGrid.from_yaml(os.path.expanduser(map_yaml))
    objects, report = consolidate(graph.objects, grid)
    output_fn(f'{path} is a raw scene graph; consolidated it:\n{report.summary()}')
    return SceneGraph(objects, graph.frame_id, consolidated=True)


_STATUS_NAMES = {
    GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
    GoalStatus.STATUS_CANCELED: 'CANCELED',
    GoalStatus.STATUS_ABORTED: 'ABORTED',
}


class RobotBrain(Node):
    def __init__(self, scene_graph, standoff=0.8, map_frame='map', base_frame='base_link',
                 image_topic='/front_stereo_camera/left/image_raw', action_name='/navigate_to_pose',
                 server_timeout=5.0, max_plan_checks=12, clearance=0.45, map_topic='/map', cmd_vel_topic='/cmd_vel',
                 preferred_clearance=0.7):
        super().__init__('robot_brain')
        self.graph = scene_graph
        self.standoff = standoff
        self.map_frame = map_frame
        self.base_frame = base_frame
        self.server_timeout = server_timeout
        self.max_plan_checks = max_plan_checks
        self.clearance = clearance        # metres a goal must keep from any occupied map cell (robot radius + margin)
        self.preferred_clearance = max(preferred_clearance, clearance)   # tried first: room to turn and leave
        self.grid = None                  # MapGrid of the latest /map, or None (then goals are only checked by Nav2)

        # Written by the spin thread, read by the chat (main) thread.
        self._lock = threading.RLock()
        self._latest_image_msg = None
        self._goal_handle = None
        self._goal_active = False
        self._goal_seq = 0          # bumped on every send/cancel so in-flight goals can be recognised as stale
        self.last_result = None
        self.result_event = threading.Event()
        self.goals_sent = 0
        self.last_goal = None       # (x, y, yaw) of the most recent goal handed to Nav2

        # Robot pose comes from the map -> base_link TF (SLAM), not from odometry.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Isaac Sim publishes sensors best-effort; a default (reliable) subscription would get nothing.
        self.image_sub = self.create_subscription(
            RosImage, image_topic, self.image_callback, qos_profile_sensor_data)
        # /map is latched (transient local) by map_server / slam_toolbox; the saved map is what Nav2 plans on too.
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self._on_map,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # STOP also zeroes the wheels directly: it works even if Nav2 is wedged and skips the cancel round trip.
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.nav_client = ActionClient(self, NavigateToPose, action_name)
        self.plan_client = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')

    # ------------------------------------------------------------------ sensing

    def image_callback(self, msg):
        with self._lock:
            self._latest_image_msg = msg

    def _on_map(self, msg):
        try:
            grid = MapGrid.from_occupancy_grid(msg)
        except ValueError as e:
            self.get_logger().warn(f'Ignoring unusable /map: {e}')
            return
        with self._lock:
            self.grid = grid

    def get_latest_image(self, max_side=1024):
        """Latest camera frame as a PIL image (downscaled for the VLM), or None."""
        with self._lock:
            msg = self._latest_image_msg
        if msg is None:
            return None
        try:
            img = PILImage.fromarray(image_msg_to_rgb(msg))
        except ValueError as e:
            self.get_logger().warn(f'Cannot decode camera image: {e}')
            return None
        img.thumbnail((max_side, max_side))
        return img

    def get_pose(self):
        """(x, y, yaw) of base_link in the map frame, or None if TF is not available yet."""
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'No {self.map_frame}->{self.base_frame} transform yet: {e}',
                                   throttle_duration_sec=5.0)
            return None
        p, q = t.transform.translation, t.transform.rotation
        return p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------------ navigation

    def cancel_current_goal_if_active(self):
        """Cancel the active goal, and any goal that was sent but not yet accepted by Nav2."""
        with self._lock:
            self._goal_seq += 1
            handle, active = self._goal_handle, self._goal_active
            self._goal_handle = None
            self._goal_active = False
        if handle is not None and active:
            self.get_logger().info('Cancelling active goal...')
            handle.cancel_goal_async()

    def send_goal(self, x, y, yaw):
        """Send a NavigateToPose goal in the map frame. Returns False if Nav2 is unreachable."""
        self.cancel_current_goal_if_active()
        if not self.nav_client.wait_for_server(timeout_sec=self.server_timeout):
            self.get_logger().error(
                f'Nav2 action server not available after {self.server_timeout:.0f}s. '
                'Is navigation_launch.py running?')
            return False
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.map_frame
        # Stamp 0 means "latest transform", which keeps working whether or not this node uses sim time.
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        (goal_msg.pose.pose.orientation.x, goal_msg.pose.pose.orientation.y,
         goal_msg.pose.pose.orientation.z, goal_msg.pose.pose.orientation.w) = yaw_to_quat(yaw)
        with self._lock:
            self._goal_seq += 1
            seq = self._goal_seq
            self.last_result = None
            self.result_event.clear()
            self.goals_sent += 1
            self.last_goal = (float(x), float(y), float(yaw))
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f, s=seq: self._goal_response_callback(f, s))
        self.get_logger().info(f'Sending goal: ({x:.2f}, {y:.2f}) facing {yaw:.2f} rad in {self.map_frame}')
        return True

    def _goal_response_callback(self, future, seq):
        try:
            handle = future.result()
        except Exception as e:      # the action server went away between the request and its answer
            self.get_logger().error(f'Goal request failed: {e}')
            with self._lock:
                if seq == self._goal_seq:
                    self.last_result = 'ERROR'
                    self.result_event.set()
            return
        with self._lock:
            stale = seq != self._goal_seq
        if not handle.accepted:
            if not stale:
                self.get_logger().warn('Goal rejected by Nav2')
                with self._lock:
                    self.last_result = 'REJECTED'
                    self.result_event.set()
            return
        if stale:
            # STOP (or a newer goal) arrived while this goal was still on its way to Nav2.
            self.get_logger().info('Cancelling goal that was superseded before Nav2 accepted it')
            handle.cancel_goal_async()
            return
        with self._lock:
            self._goal_handle = handle
            self._goal_active = True
        handle.get_result_async().add_done_callback(lambda f, h=handle: self._result_callback(f, h))

    def _result_callback(self, future, handle):
        status = future.result().status
        name = _STATUS_NAMES.get(status, f'STATUS_{status}')
        with self._lock:
            if handle is self._goal_handle:      # ignore results of goals we already replaced
                self._goal_active = False
                self._goal_handle = None
                self.last_result = name
                self.result_event.set()
        # (rclpy forbids logging at different severities from one call site, so keep two explicit calls)
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation finished: SUCCEEDED')
        else:
            self.get_logger().warn(f'Navigation finished: {name}')

    def _wait(self, future, timeout):
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            time.sleep(0.02)
        return future.done()

    def has_path(self, x, y):
        """Ask Nav2's planner whether a path to (x, y) exists. True when it cannot be asked (never blocks a goal)."""
        if not self.plan_client.wait_for_server(timeout_sec=1.0):
            return True
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = self.map_frame
        goal.goal.pose.position.x, goal.goal.pose.position.y, goal.goal.pose.orientation.w = float(x), float(y), 1.0
        goal.use_start = False
        sent = self.plan_client.send_goal_async(goal)
        if not self._wait(sent, 3.0):
            return True
        handle = sent.result()
        if not handle.accepted:
            return False
        result = handle.get_result_async()
        if not self._wait(result, 8.0):
            return True
        res = result.result()
        return res.status == GoalStatus.STATUS_SUCCEEDED and len(res.result.path.poses) >= 1

    def approach_points(self, obj, robot_xy, distance):
        """Stop points `distance` m from the object (measured from its edge for a group), best first.

        Without a map: just the point on the line from the object to the robot. With the map: also points around
        the object, keeping only known free cells with clearance from walls and a clear line to the object, so the
        planner is not asked about spots inside walls, behind them, or in unexplored space.
        """
        distance += obj.radius
        base = standoff_point(obj.xy, robot_xy, distance)
        with self._lock:
            grid = self.grid
        if grid is None or base == tuple(robot_xy):
            return [base]           # (already close enough: the robot only turns to face the object)
        usable = [p for p in ring_points(obj.xy, robot_xy, distance)
                  if grid.is_free(p[0], p[1], self.clearance) and grid.segment_is_free(p, obj.xy, ignore_end=0.4)]
        roomy = [p for p in usable if grid.is_free(p[0], p[1], self.preferred_clearance)]
        # Roomy stop points first (the robot can turn and leave without touching anything the lidar cannot see),
        # then the merely legal ones.
        return (roomy[:3] + [p for p in usable if p not in roomy][:2])[:4]

    def choose_approach(self, name, robot_xy):
        """Pick (object, goal_x, goal_y) for a named goal, or None if Nav2 finds no path to any approach point.

        Instances are tried nearest first (a bare label means "any"), each at the stand-off distance and then
        a bit further out: a point 0.8 m from a surface can sit inside the obstacle's inflated zone or behind
        depth error, while 1.2-2.4 m out is usually reachable.
        """
        candidates = self.graph.resolve_all(name, robot_xy)
        if not candidates:
            return 'unknown'
        checked = 0
        for obj in candidates:
            for extra in (0.0, 0.4, 0.8, 1.6):
                for gx, gy in self.approach_points(obj, robot_xy, self.standoff + extra):
                    if self.max_plan_checks <= 0:      # probing disabled: nearest instance at the plain stand-off
                        return obj, gx, gy
                    if math.hypot(gx - robot_xy[0], gy - robot_xy[1]) < 0.3:
                        return obj, gx, gy             # already there: nothing to plan
                    if checked >= self.max_plan_checks:
                        return None
                    checked += 1
                    if self.has_path(gx, gy):
                        return obj, gx, gy
        return None

    def stop_robot(self):
        self.get_logger().info('STOP requested')
        self.cancel_current_goal_if_active()
        self.cmd_vel_pub.publish(Twist())      # all-zero velocity

    # ------------------------------------------------------------------ command dispatch

    def execute(self, cmd):
        """Carry out a parsed Command. Returns a short message for the user, or None."""
        if cmd.kind == 'STOP':
            self.stop_robot()
            return 'Stopping.'
        if cmd.kind not in ('GOAL', 'RELATIVE'):
            return None

        pose = self.get_pose()
        if pose is None:
            return (f'Cannot navigate: no {self.map_frame}->{self.base_frame} transform. '
                    'Is the SLAM pipeline running?')
        x, y, yaw = pose

        if cmd.kind == 'GOAL':
            choice = self.choose_approach(cmd.target, (x, y))
            if choice == 'unknown':
                return f"Unknown object '{cmd.target}'. Known: {', '.join(self.graph.names()) or '(none)'}."
            if choice is None:
                return (f"Nav2 finds no path to any approach point of '{cmd.target}' from here "
                        '(blocked, or outside the mapped area). Not moving.')
            obj, gx, gy = choice
            gyaw = yaw_facing((gx, gy), obj.xy)
            ok = self.send_goal(gx, gy, gyaw)
            what = f"'{obj.id}' at ({obj.x:.2f}, {obj.y:.2f}), stopping at ({gx:.2f}, {gy:.2f})"
        else:
            gx, gy = rotate_relative_offset(x, y, yaw, cmd.dx, cmd.dy)
            shortened = ''
            with self._lock:
                grid = self.grid
            if grid is not None and grid.state(x, y) == FREE and not grid.is_free(gx, gy, self.clearance):
                # The point asked for is inside a wall / unexplored space: go as far as the free floor allows instead
                # of sending Nav2 a goal it can only abort.
                stop = grid.last_free_point((x, y), (gx, gy), self.clearance)
                if stop is None or math.hypot(stop[0] - x, stop[1] - y) < 0.3:
                    return (f'Cannot move ({cmd.dx:+.1f}, {cmd.dy:+.1f}) m from here: something (a wall or '
                            'unexplored space) is in the way. Not moving.')
                shortened = f' (shortened from {math.hypot(cmd.dx, cmd.dy):.1f} m: the rest is blocked)'
                gx, gy = stop
            ok = self.send_goal(gx, gy, yaw)
            what = f'relative move ({cmd.dx:+.2f}, {cmd.dy:+.2f}) -> ({gx:.2f}, {gy:.2f}){shortened}'
        return f'Going to {what}.' if ok else 'Nav2 is not available; goal not sent.'


class QwenChat:
    """Thin wrapper around Qwen2.5-VL: (system prompt, user text, optional PIL image) -> reply text."""

    def __init__(self, model_id='Qwen/Qwen2.5-VL-3B-Instruct', max_new_tokens=150):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype='auto', device_map='auto')
        self.processor = AutoProcessor.from_pretrained(model_id)

    def __call__(self, system_prompt, user_text, image=None):
        if image is not None:
            user = {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                {'type': 'text', 'text': user_text}]}
        else:
            user = {'role': 'user', 'content': user_text}
        messages = [{'role': 'system', 'content': system_prompt}, user]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {'images': [image]} if image is not None else {}
        inputs = self.processor(text=[text], return_tensors='pt', **kwargs).to(self.model.device)
        with self._torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        reply = self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return first_command_line(reply)


def chat_loop(robot, llm, input_fn=input, output_fn=print, wait_for_result=False, result_timeout=300.0):
    """Read user lines, ask the model, and dispatch its command to the robot.

    A failing model call or command never ends the session. With `wait_for_result` the loop blocks after each goal
    until Nav2 reports back (for scripted runs, where leaving would cancel the goal).
    """
    while True:
        try:
            user_input = input_fn('You: ').strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in ('exit', 'quit'):
            break

        question = is_question(user_input)
        pose = robot.get_pose()
        system_prompt = (build_question_prompt if question else build_system_prompt)(
            robot.graph, pose, mentioned_objects(user_input, robot.graph))
        image = robot.get_latest_image() if needs_vision(user_input) else None
        try:
            response = llm(system_prompt, user_input, image)
        except Exception as e:      # e.g. CUDA out of memory: report it and keep the session alive
            output_fn(f'\n[ERROR] The language model failed ({type(e).__name__}: {e}). Nothing was sent to the robot.\n')
            continue
        output_fn(f'\nRobot: {response}\n')
        if question:                    # a question is never an order: do not act on whatever the model replied
            continue

        sent_before = robot.goals_sent
        try:
            cmd = reconcile_goal(parse_response(response, robot.graph.names()), user_input, robot.graph)
            note = robot.execute(cmd)
        except Exception as e:
            output_fn(f'[ERROR] Could not carry out the command ({type(e).__name__}: {e}).')
            continue
        if note:
            output_fn(f'[NAV] {note}')
        if wait_for_result and robot.goals_sent > sent_before:
            finished = robot.result_event.wait(result_timeout)
            output_fn(f'[NAV] Result: {robot.last_result if finished else "TIMEOUT (still moving)"}')


def parse_args(argv):
    p = argparse.ArgumentParser(description='Natural-language robot brain (Qwen2.5-VL + Nav2).')
    p.add_argument('--scene-graph', default=os.path.join(os.path.expanduser('~'), 'scene_graph', 'scene_graph.json'))
    p.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--standoff', type=float, default=0.8, help='metres to stop short of a named object')
    p.add_argument('--map-frame', default='map')
    p.add_argument('--base-frame', default='base_link')
    p.add_argument('--image-topic', default='/front_stereo_camera/left/image_raw')
    p.add_argument('--action-name', default='/navigate_to_pose')
    p.add_argument('--server-timeout', type=float, default=5.0, help='seconds to wait for Nav2')
    p.add_argument('--max-plan-checks', type=int, default=12, help='planner queries allowed when choosing an approach point')
    p.add_argument('--clearance', type=float, default=0.45,
                   help='metres a goal must keep from walls on the map (robot radius + margin)')
    p.add_argument('--preferred-clearance', type=float, default=0.7,
                   help='metres from walls a stop point should have when one exists (else --clearance is enough)')
    p.add_argument('--map-topic', default='/map')
    p.add_argument('--map-yaml', default=os.path.join(os.path.expanduser('~'), 'my_map.yaml'),
                   help='saved map, used only to judge a raw scene graph when consolidating it (ignored if missing)')
    p.add_argument('--no-consolidate', action='store_true', help='use the scene graph exactly as saved')
    p.add_argument('--cmd-vel-topic', default='/cmd_vel', help='where STOP publishes a zero velocity')
    p.add_argument('--wait-result', action='store_true',
                   help='after each goal wait for Nav2 to finish before the next line (implied when stdin is not a terminal)')
    return p.parse_args(argv)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    rclpy.init(args=argv)
    args = parse_args(rclpy.utilities.remove_ros_args(argv)[1:])

    graph = load_graph(args.scene_graph, not args.no_consolidate, args.map_yaml)
    print(f'Loaded {len(graph)} objects from {args.scene_graph}: {", ".join(o.id for o in graph.objects) or "(none)"}')

    robot = RobotBrain(graph, args.standoff, args.map_frame, args.base_frame, args.image_topic,
                       args.action_name, args.server_timeout, args.max_plan_checks, args.clearance, args.map_topic,
                       args.cmd_vel_topic, args.preferred_clearance)
    spinner = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    spinner.start()
    try:
        print('Loading Qwen2.5-VL...')
        llm = QwenChat(args.model)
        print('\nRobot ready! Type your command (exit/quit to stop)\n')
        chat_loop(robot, llm, wait_for_result=args.wait_result or not sys.stdin.isatty())
    finally:
        robot.cancel_current_goal_if_active()
        # Stop the spin thread *before* tearing the node down; exiting with it still inside rclpy aborts the process.
        rclpy.shutdown()
        spinner.join(timeout=3.0)
        robot.destroy_node()


if __name__ == '__main__':
    main()
