import json
import re
import threading
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as RosImage
from nav2_msgs.action import NavigateToPose
from PIL import Image as PILImage
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

SCENE_GRAPH_PATH = "/home/user/scene_graph/scene_graph.json"
OBJECT_OFFSET = 0.6

with open(SCENE_GRAPH_PATH, 'r') as f:
    scene_graph = json.load(f)


class RobotBrain(Node):
    def __init__(self):
        super().__init__('robot_brain')
        self.current_pose = {'x': 0.0, 'y': 0.0}
        self.latest_image = None
        self.odom_sub = self.create_subscription(
            Odometry, '/chassis/odom', self.odom_callback, 10
        )
        self.image_sub = self.create_subscription(
            RosImage, '/front_stereo_camera/left/image_raw', self.image_callback, 10
        )
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._goal_handle = None
        self._goal_active = False

    def odom_callback(self, msg):
        self.current_pose['x'] = msg.pose.pose.position.x
        self.current_pose['y'] = msg.pose.pose.position.y

    def image_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.latest_image = PILImage.fromarray(img)

    def cancel_current_goal_if_active(self):
        if self._goal_handle is not None and self._goal_active:
            self.get_logger().info('Cancelling active goal...')
            self._goal_handle.cancel_goal_async()
        self._goal_active = False
        self._goal_handle = None

    def send_goal(self, x, y):
        self.cancel_current_goal_if_active()
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0
        self.nav_client.wait_for_server()
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)
        self.get_logger().info(f'Sending goal: ({x:.2f}, {y:.2f})')

    def _goal_response_callback(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected')
            self._goal_active = False
            return
        self._goal_handle = handle
        self._goal_active = True
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        self._goal_active = False

    def stop_robot(self):
        self.get_logger().info('STOP requested')
        self.cancel_current_goal_if_active()


def offset_toward_robot(obj_x, obj_y, robot_x, robot_y, offset=OBJECT_OFFSET):
    dx = robot_x - obj_x
    dy = robot_y - obj_y
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 1e-3:
        return obj_x, obj_y
    ratio = offset / dist
    return obj_x + dx * ratio, obj_y + dy * ratio


def build_system_prompt(current_x=0.0, current_y=0.0):
    ctx = "You are a robot navigation assistant. You control a mobile robot with a front camera.\n\n"
    ctx += f"The robot's CURRENT position right now is (x={current_x:.2f}, y={current_y:.2f}).\n"
    ctx += "Always use this current position when asked where you are, or when computing relative moves.\n\n"
    ctx += "Known objects in the environment (semantic map):\n"
    for label, data in scene_graph.items():
        ctx += f"- {label}: position (x={data['x']}, y={data['y']})\n"
    ctx += "\nRespond using ONLY one of these formats:\n"
    ctx += "1. GOAL:(x,y) - use ONLY when navigating to a named object from the semantic map above (use that object's exact coordinates)\n"
    ctx += "2. RELATIVE:(dx,dy) - use for ANY movement described relative to the robot itself, e.g. 'go forward 1 meter', "
    ctx += "'move back 2 meters', 'go left 1 meter'. dx is forward(+)/backward(-) offset in meters, dy is left(+)/right(-) offset in meters. "
    ctx += "Do NOT compute absolute coordinates yourself for these - just output the offset (dx,dy), the system will add it to the current position.\n"
    ctx += "3. STOP - stop the robot\n"
    ctx += "4. Plain text for questions/descriptions, including describing what the camera sees\n"
    ctx += "If an image is provided and the user asks what you see, describe it briefly in 2-3 lines.\n"
    ctx += "Always add a short explanation after any command on a new line.\n"
    return ctx


def parse_response(response, current_x, current_y):
    goal_match = re.search(r'GOAL:\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)', response)
    rel_match = re.search(r'RELATIVE:\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)', response)
    stop_match = re.search(r'\bSTOP\b', response)

    if stop_match:
        return ('STOP', None, None)
    if goal_match:
        return ('GOAL', float(goal_match.group(1)), float(goal_match.group(2)))
    if rel_match:
        dx, dy = float(rel_match.group(1)), float(rel_match.group(2))
        return ('RELATIVE', current_x + dx, current_y + dy)
    return (None, None, None)


def find_nearest_object(x, y):
    best_label, best_dist = None, float('inf')
    for label, data in scene_graph.items():
        d = math.sqrt((data['x'] - x)**2 + (data['y'] - y)**2)
        if d < best_dist:
            best_dist = d
            best_label = label
    return best_label, best_dist


VISION_TRIGGERS = ['see', 'look', 'camera', 'view', 'around you', 'in front']


def needs_vision(user_input):
    text = user_input.lower()
    return any(trigger in text for trigger in VISION_TRIGGERS)


def main():
    rclpy.init()
    robot = RobotBrain()
    spinner = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    spinner.start()

    print("Loading Qwen2.5-VL 3B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype='auto', device_map='auto'
    )
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')

    print("\nRobot ready! Type your command (exit/quit to stop)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break

        system_prompt = build_system_prompt(robot.current_pose['x'], robot.current_pose['y'])
        use_vision = needs_vision(user_input) and robot.latest_image is not None

        if use_vision:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": robot.latest_image},
                    {"type": "text", "text": user_input}
                ]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[robot.latest_image], return_tensors="pt").to(model.device)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=150, do_sample=False)

        response = processor.decode(
            output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )
        print(f"\nRobot: {response}\n")

        action, x, y = parse_response(response, robot.current_pose['x'], robot.current_pose['y'])

        if action == 'STOP':
            robot.stop_robot()
        elif action == 'GOAL':
            label, dist = find_nearest_object(x, y)
            if label and dist < 0.3:
                ox, oy = offset_toward_robot(x, y, robot.current_pose['x'], robot.current_pose['y'])
                print(f"[NAV] Offsetting goal away from '{label}' -> ({ox:.2f},{oy:.2f})")
                robot.send_goal(ox, oy)
            else:
                robot.send_goal(x, y)
        elif action == 'RELATIVE':
            print(f"[NAV] Relative move -> ({x:.2f},{y:.2f})")
            robot.send_goal(x, y)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
