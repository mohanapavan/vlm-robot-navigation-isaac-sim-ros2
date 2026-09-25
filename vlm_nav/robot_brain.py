"""Qwen2.5-VL natural-language robot brain: chat -> GOAL / RELATIVE / STOP -> Nav2.

Run (Nav2 + the SLAM pipeline must already be up, see docs/commands.md):
    ros2 run vlm_nav robot_brain --scene-graph ~/scene_graph/scene_graph.json --ros-args -p use_sim_time:=true
"""
import argparse
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
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image as RosImage

from .command_parser import (build_question_prompt, build_system_prompt, first_command_line, is_question,
                             needs_vision, parse_response)
from .geometry import quat_to_yaw, rotate_relative_offset, standoff_point, yaw_facing, yaw_to_quat
from .image_utils import image_msg_to_rgb
from .scene_graph import SceneGraph

_STATUS_NAMES = {
    GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
    GoalStatus.STATUS_CANCELED: 'CANCELED',
    GoalStatus.STATUS_ABORTED: 'ABORTED',
}


class RobotBrain(Node):
    def __init__(self, scene_graph, standoff=0.8, map_frame='map', base_frame='base_link',
                 image_topic='/front_stereo_camera/left/image_raw', action_name='/navigate_to_pose',
                 server_timeout=5.0, max_plan_checks=12):
        super().__init__('robot_brain')
        self.graph = scene_graph
        self.standoff = standoff
        self.map_frame = map_frame
        self.base_frame = base_frame
        self.server_timeout = server_timeout
        self.max_plan_checks = max_plan_checks

        # Written by the spin thread, read by the chat (main) thread.
        self._lock = threading.RLock()
        self._latest_image_msg = None
        self._goal_handle = None
        self._goal_active = False
        self._goal_seq = 0          # bumped on every send/cancel so in-flight goals can be recognised as stale
        self.last_result = None
        self.result_event = threading.Event()

        # Robot pose comes from the map -> base_link TF (SLAM), not from odometry.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Isaac Sim publishes sensors best-effort; a default (reliable) subscription would get nothing.
        self.image_sub = self.create_subscription(
            RosImage, image_topic, self.image_callback, qos_profile_sensor_data)
        self.nav_client = ActionClient(self, NavigateToPose, action_name)
        self.plan_client = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')

    # ------------------------------------------------------------------ sensing

    def image_callback(self, msg):
        with self._lock:
            self._latest_image_msg = msg

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
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f, s=seq: self._goal_response_callback(f, s))
        self.get_logger().info(f'Sending goal: ({x:.2f}, {y:.2f}) facing {yaw:.2f} rad in {self.map_frame}')
        return True

    def _goal_response_callback(self, future, seq):
        handle = future.result()
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
        return res.status == GoalStatus.STATUS_SUCCEEDED and len(res.result.path.poses) > 1

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
                gx, gy = standoff_point(obj.xy, robot_xy, self.standoff + extra)
                if self.max_plan_checks <= 0:          # probing disabled: nearest instance at the plain stand-off
                    return obj, gx, gy
                if checked >= self.max_plan_checks:
                    return None
                checked += 1
                if self.has_path(gx, gy):
                    return obj, gx, gy
        return None

    def stop_robot(self):
        self.get_logger().info('STOP requested')
        self.cancel_current_goal_if_active()

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
            ok = self.send_goal(gx, gy, yaw)
            what = f'relative move ({cmd.dx:+.2f}, {cmd.dy:+.2f}) -> ({gx:.2f}, {gy:.2f})'
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


def chat_loop(robot, llm, input_fn=input, output_fn=print):
    """Read user lines, ask the model, and dispatch its command to the robot."""
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
        system_prompt = (build_question_prompt if question else build_system_prompt)(robot.graph, pose)
        image = robot.get_latest_image() if needs_vision(user_input) else None
        response = llm(system_prompt, user_input, image)
        output_fn(f'\nRobot: {response}\n')
        if question:                    # a question is never an order: do not act on whatever the model replied
            continue

        note = robot.execute(parse_response(response, robot.graph.names()))
        if note:
            output_fn(f'[NAV] {note}')


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
    return p.parse_args(argv)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    rclpy.init(args=argv)
    args = parse_args(rclpy.utilities.remove_ros_args(argv)[1:])

    graph = SceneGraph.load(args.scene_graph)
    print(f'Loaded {len(graph)} objects from {args.scene_graph}: {", ".join(o.id for o in graph.objects) or "(none)"}')

    robot = RobotBrain(graph, args.standoff, args.map_frame, args.base_frame, args.image_topic,
                       args.action_name, args.server_timeout, args.max_plan_checks)
    spinner = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    spinner.start()
    try:
        print('Loading Qwen2.5-VL...')
        llm = QwenChat(args.model)
        print('\nRobot ready! Type your command (exit/quit to stop)\n')
        chat_loop(robot, llm)
    finally:
        robot.cancel_current_goal_if_active()
        # Stop the spin thread *before* tearing the node down; exiting with it still inside rclpy aborts the process.
        rclpy.shutdown()
        spinner.join(timeout=3.0)
        robot.destroy_node()


if __name__ == '__main__':
    main()
