"""ROS-level tests for RobotBrain against a fake Nav2 action server and fake TF.

No Isaac Sim, Nav2 or GPU is needed, only a sourced ROS 2 Humble. They run on their own DDS domain so
they never mix with a running simulator.
"""
import os

os.environ.setdefault('ROS_DOMAIN_ID', '87')

import math  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('nav2_msgs')
pytest.importorskip('tf2_ros')

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist  # noqa: E402
from nav2_msgs.action import ComputePathToPose, NavigateToPose  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402

from vlm_nav.command_parser import Command  # noqa: E402
from vlm_nav.geometry import quat_to_yaw, yaw_to_quat  # noqa: E402
from vlm_nav.map_grid import OCCUPIED, UNKNOWN, MapGrid  # noqa: E402
from vlm_nav.robot_brain import RobotBrain, chat_loop  # noqa: E402
from vlm_nav.scene_graph import SceneGraph, SceneObject  # noqa: E402


def wait_for(cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


class FakeWorld(Node):
    """Publishes map->base_link TF and camera images, and serves /navigate_to_pose."""

    def __init__(self):
        super().__init__('fake_world')
        self.pose = (2.0, 3.0, math.pi / 2)
        self.publish_tf = True
        self.goals = []
        self.mode = 'succeed'          # 'succeed' | 'abort' | 'hold' (run until cancelled)
        self.cancelled = 0
        self.blocked = []              # (x, y, radius) regions the fake planner cannot plan into
        self.plan_queries = []
        self.cmd_vels = []
        self.create_subscription(Twist, '/cmd_vel', lambda m: self.cmd_vels.append(m), 10)
        self.tf = TransformBroadcaster(self)
        self.create_timer(0.05, self._pub_tf)
        self.image_pub = self.create_publisher(Image, '/front_stereo_camera/left/image_raw', qos_profile_sensor_data)
        self.server = ActionServer(
            self, NavigateToPose, 'navigate_to_pose', execute_callback=self._execute,
            goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=lambda h: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup())

        self.plan_server = ActionServer(
            self, ComputePathToPose, 'compute_path_to_pose', execute_callback=self._plan,
            goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=lambda h: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup())

    def _plan(self, goal_handle):
        p = goal_handle.request.goal.pose.position
        self.plan_queries.append((p.x, p.y))
        result = ComputePathToPose.Result()
        if any(math.hypot(p.x - bx, p.y - by) <= r for bx, by, r in self.blocked):
            goal_handle.abort()
        else:
            result.path.poses = [PoseStamped() for _ in range(10)]
            goal_handle.succeed()
        return result

    def _pub_tf(self):
        if not self.publish_tf:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = 'map', 'base_link'
        t.transform.translation.x, t.transform.translation.y = self.pose[0], self.pose[1]
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = yaw_to_quat(self.pose[2])
        self.tf.sendTransform(t)

    def _execute(self, goal_handle):
        self.goals.append(goal_handle.request.pose)
        mode = self.mode
        if mode == 'hold':
            while not goal_handle.is_cancel_requested:
                time.sleep(0.02)
            self.cancelled += 1
            goal_handle.canceled()
        elif mode == 'abort':
            time.sleep(0.1)
            goal_handle.abort()
        else:
            time.sleep(0.1)
            goal_handle.succeed()
        return NavigateToPose.Result()

    def publish_image(self, arr, encoding):
        m = Image()
        m.header.frame_id = 'cam'
        m.height, m.width = arr.shape[:2]
        m.encoding, m.step, m.is_bigendian = encoding, arr.shape[1] * arr.shape[2], 0
        m.data = arr.tobytes()
        self.image_pub.publish(m)


GRAPH = SceneGraph([
    SceneObject('forklift_1', 'forklift', 10.0, 3.0, 0.5, 4, 0.7),
    SceneObject('shelf_1', 'shelf', 2.0, 10.0, 1.0, 3, 0.6),
    SceneObject('box_1', 'box', -4.0, 3.0, 0.2, 5, 0.5),
    SceneObject('box_2', 'box', 6.0, 3.5, 0.2, 5, 0.5),
])


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    world = FakeWorld()
    brain = RobotBrain(GRAPH, standoff=0.8, server_timeout=3.0)
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(world)
    ex.add_node(brain)
    t = threading.Thread(target=ex.spin, daemon=True)
    t.start()
    assert wait_for(lambda: brain.get_pose() is not None), 'no map->base_link TF reached the brain'
    yield world, brain
    ex.shutdown()
    brain.destroy_node()
    world.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def env(ros):
    world, brain = ros
    brain.cancel_current_goal_if_active()
    world.goals.clear()
    world.blocked, world.plan_queries, world.cmd_vels = [], [], []
    brain.grid = None
    world.mode, world.cancelled, world.pose, world.publish_tf = 'succeed', 0, (2.0, 3.0, math.pi / 2), True
    wait_for(lambda: brain.get_pose() is not None)
    time.sleep(0.15)
    world.goals.clear()
    return world, brain


def goal_yaw(pose):
    q = pose.pose.orientation
    return quat_to_yaw(q.x, q.y, q.z, q.w)


def test_pose_comes_from_map_to_base_link_tf(env):
    world, brain = env
    world.pose = (5.0, -1.0, 0.5)
    assert wait_for(lambda: brain.get_pose() and abs(brain.get_pose()[0] - 5.0) < 1e-6)
    x, y, yaw = brain.get_pose()
    assert (x, y, yaw) == pytest.approx((5.0, -1.0, 0.5), abs=1e-6)


def test_goal_is_in_map_frame_with_standoff_and_faces_the_object(env):
    world, brain = env
    msg = brain.execute(Command('GOAL', target='forklift'))
    assert wait_for(lambda: len(world.goals) == 1)
    g = world.goals[0]
    assert g.header.frame_id == 'map'
    # robot at (2,3); forklift at (10,3) -> stop 0.8 m short on the robot's side, facing +x toward it
    assert (g.pose.position.x, g.pose.position.y) == pytest.approx((9.2, 3.0), abs=1e-6)
    assert goal_yaw(g) == pytest.approx(0.0, abs=1e-6)
    assert 'forklift_1' in msg


def test_goal_by_bare_label_picks_nearest_instance(env):
    world, brain = env
    brain.execute(Command('GOAL', target='box'))            # box_1 is 6 m away, box_2 is 4.03 m away
    assert wait_for(lambda: len(world.goals) == 1)
    assert world.goals[0].pose.position.x == pytest.approx(5.2, abs=0.05)


def test_relative_move_uses_robot_heading(env):
    world, brain = env                                       # robot at (2,3) facing +y
    brain.execute(Command('RELATIVE', dx=1.0, dy=0.0))      # 1 m forward -> +y
    assert wait_for(lambda: len(world.goals) == 1)
    g = world.goals[0]
    assert (g.pose.position.x, g.pose.position.y) == pytest.approx((2.0, 4.0), abs=1e-6)
    assert goal_yaw(g) == pytest.approx(math.pi / 2, abs=1e-6)          # keeps its heading

    brain.execute(Command('RELATIVE', dx=0.0, dy=1.0))      # 1 m to the left of a +y-facing robot -> -x
    assert wait_for(lambda: len(world.goals) == 2)
    assert (world.goals[1].pose.position.x, world.goals[1].pose.position.y) == pytest.approx((1.0, 3.0), abs=1e-6)


def test_unknown_object_sends_nothing(env):
    world, brain = env
    msg = brain.execute(Command('GOAL', target='spaceship'))
    time.sleep(0.4)
    assert world.goals == [] and 'Unknown object' in msg and 'forklift_1' in msg


def test_no_tf_means_no_goal(env):
    world, brain = env
    world.publish_tf = False
    time.sleep(0.3)
    brain.tf_buffer.clear()
    msg = brain.execute(Command('RELATIVE', dx=1.0, dy=0.0))
    assert 'no map->base_link transform' in msg.lower() or 'no map->base_link' in msg
    assert world.goals == []


def test_success_result_is_reported(env):
    world, brain = env
    brain.execute(Command('RELATIVE', dx=1.0, dy=0.0))
    assert brain.result_event.wait(6.0)
    assert brain.last_result == 'SUCCEEDED'


def test_abort_result_is_reported(env):
    world, brain = env
    world.mode = 'abort'
    brain.execute(Command('RELATIVE', dx=1.0, dy=0.0))
    assert brain.result_event.wait(6.0)
    assert brain.last_result == 'ABORTED'


def test_stop_cancels_the_active_goal(env):
    world, brain = env
    world.mode = 'hold'
    brain.execute(Command('GOAL', target='forklift'))
    assert wait_for(lambda: len(world.goals) == 1 and brain._goal_active)
    assert brain.execute(Command('STOP')) == 'Stopping.'
    assert wait_for(lambda: world.cancelled == 1)
    assert not brain._goal_active


def test_new_goal_replaces_the_active_one(env):
    world, brain = env
    world.mode = 'hold'
    brain.execute(Command('GOAL', target='forklift'))
    assert wait_for(lambda: len(world.goals) == 1 and brain._goal_active)
    world.mode = 'succeed'
    brain.execute(Command('RELATIVE', dx=1.0, dy=0.0))
    assert wait_for(lambda: world.cancelled == 1)             # the first goal was cancelled...
    assert brain.result_event.wait(6.0)
    assert brain.last_result == 'SUCCEEDED'                   # ...and its late CANCELED result did not clobber ours


def test_missing_nav2_times_out_instead_of_blocking(ros):
    _, brain = ros
    other = RobotBrain(GRAPH, server_timeout=0.5)
    other.nav_client.destroy()
    from rclpy.action import ActionClient
    other.nav_client = ActionClient(other, NavigateToPose, '/no_such_nav2_server')
    t0 = time.time()
    assert other.send_goal(1.0, 1.0, 0.0) is False
    assert time.time() - t0 < 3.0
    other.destroy_node()


def test_camera_frames_arrive_with_sensor_data_qos(env):
    """Isaac Sim publishes best-effort. A default (reliable) subscription would receive nothing."""
    world, brain = env
    default_qos_msgs = []
    probe = world.create_subscription(Image, '/front_stereo_camera/left/image_raw',
                                      lambda m: default_qos_msgs.append(m), 10)
    frame = np.random.default_rng(0).integers(0, 255, (48, 64, 4), dtype=np.uint8)     # rgba8
    for _ in range(40):
        world.publish_image(frame, 'rgba8')
        time.sleep(0.02)
    assert wait_for(lambda: brain.get_latest_image() is not None)
    img = brain.get_latest_image()
    assert img.size == (64, 48) and img.mode == 'RGB'
    assert np.array_equal(np.asarray(img), frame[:, :, :3])
    assert default_qos_msgs == []                              # what the old subscription would have seen
    world.destroy_subscription(probe)


def test_image_access_is_thread_safe(env):
    world, brain = env
    frame = np.zeros((32, 32, 3), np.uint8)
    errors = []

    def reader():
        try:
            for _ in range(150):
                brain.get_latest_image()
        except Exception as e:   # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for _ in range(150):
        world.publish_image(frame, 'rgb8')
        time.sleep(0.005)
    for t in threads:
        t.join()
    assert errors == []


def _run_chat(brain, replies, lines):
    prompts, it = [], iter(lines)

    def llm(system_prompt, user_text, image):
        prompts.append((system_prompt, user_text, image))
        return replies.pop(0)

    out = []
    chat_loop(brain, llm, input_fn=lambda _: next(it), output_fn=out.append)
    return prompts, out


def test_chat_goal_with_explanation_mentioning_stop_still_moves(env):
    world, brain = env
    prompts, out = _run_chat(brain, ["GOAL:forklift\nI'll stop at the shelf on the way."],
                             ['go to the forklift', 'exit'])
    assert wait_for(lambda: len(world.goals) == 1)
    assert world.goals[0].pose.position.x == pytest.approx(9.2, abs=1e-6)
    assert '10.0' not in prompts[0][0] and 'x=10' not in prompts[0][0]        # coordinates never shown to the model
    assert 'forklift_1' in prompts[0][0]


def test_chat_stop_cancels(env):
    world, brain = env
    world.mode = 'hold'
    _run_chat(brain, ['GOAL:shelf_1\nok', 'STOP\nStopping.'], ['go to the shelf', 'stop', 'quit'])
    assert wait_for(lambda: world.cancelled == 1)


def test_chat_vision_prompt_passes_camera_image(env):
    world, brain = env
    frame = np.full((24, 32, 3), 100, np.uint8)
    for _ in range(30):
        world.publish_image(frame, 'rgb8')
        time.sleep(0.02)
    assert wait_for(lambda: brain.get_latest_image() is not None)
    prompts, _ = _run_chat(brain, ['I see a wall.'], ['what do you see?', 'exit'])
    assert prompts[0][2] is not None and prompts[0][2].size == (32, 24)


def test_stop_right_after_sending_a_goal_still_cancels_it(env):
    """STOP can arrive before Nav2 has even accepted the goal; the goal must not survive."""
    world, brain = env
    world.mode = 'hold'
    brain.execute(Command('GOAL', target='forklift'))
    brain.execute(Command('STOP'))                          # immediately, goal response still in flight
    assert wait_for(lambda: world.cancelled == 1)
    assert not brain._goal_active


def test_chat_question_is_answered_but_never_acted_on(env):
    """Even if the model answers a question with a GOAL, the robot must not move."""
    world, brain = env
    prompts, out = _run_chat(brain, ['GOAL:forklift_1'], ['which forklift is closest to you?', 'exit'])
    time.sleep(0.6)
    assert world.goals == []
    assert 'GOAL' not in prompts[0][0] and 'Do not output commands' in prompts[0][0]
    assert any('Robot: GOAL:forklift_1' in o for o in out) and not any('[NAV]' in o for o in out)


def test_unreachable_nearest_instance_falls_back_to_the_next_one(env):
    world, brain = env                                       # robot (2,3); box_2 (6,3.5) is nearest, box_1 (-4,3) next
    world.blocked = [(5.2, 3.4, 3.0)]                        # nothing near box_2 can be planned to, at any stand-off
    msg = brain.execute(Command('GOAL', target='box'))
    assert wait_for(lambda: len(world.goals) == 1)
    assert world.goals[0].pose.position.x == pytest.approx(-3.2, abs=0.05)         # box_1, 0.8 m short
    assert "'box_1'" in msg


def test_blocked_standoff_is_moved_further_from_the_object(env):
    world, brain = env                                       # forklift at (10,3): plain stand-off is (9.2, 3.0)
    world.blocked = [(9.2, 3.0, 0.2)]
    brain.execute(Command('GOAL', target='forklift'))
    assert wait_for(lambda: len(world.goals) == 1)
    assert world.goals[0].pose.position.x == pytest.approx(8.8, abs=0.05)          # 1.2 m from the object instead


def test_nothing_reachable_means_no_goal_and_a_clear_message(env):
    world, brain = env
    world.blocked = [(10.0, 3.0, 5.0)]                       # the whole forklift area
    msg = brain.execute(Command('GOAL', target='forklift_1'))
    time.sleep(0.4)
    assert world.goals == [] and 'no path' in msg and 'Not moving' in msg
    assert len(world.plan_queries) == 4                      # 4 stand-off distances tried for the single instance


def test_goal_still_sent_when_the_planner_cannot_be_asked(env):
    from rclpy.action import ActionClient
    world, brain = env
    real = brain.plan_client
    brain.plan_client = ActionClient(brain, ComputePathToPose, '/no_such_planner')
    try:
        brain.execute(Command('GOAL', target='forklift'))
        assert wait_for(lambda: len(world.goals) == 1)
        assert world.goals[0].pose.position.x == pytest.approx(9.2, abs=0.05)
    finally:
        brain.plan_client.destroy()
        brain.plan_client = real


# ---------------------------------------------------------------- map-aware behaviour, groups, resilience

def room_grid(blocks=(), unknown_above=None):
    """x in [-5, 15], y in [-2, 8] at 0.1 m, all free; `blocks` are (x0, y0, x1, y1) occupied rectangles."""
    cells = np.zeros((100, 200), dtype=np.int16)
    for x0, y0, x1, y1 in blocks:
        cells[int((y0 + 2) / 0.1):int((y1 + 2) / 0.1), int((x0 + 5) / 0.1):int((x1 + 5) / 0.1)] = OCCUPIED
    if unknown_above is not None:
        cells[int((unknown_above + 2) / 0.1):, :] = UNKNOWN
    return MapGrid(cells, 0.1, (-5.0, -2.0))


def test_approach_points_skip_walls_and_unexplored_space(env):
    world, brain = env
    forklift = GRAPH.objects[0]                                   # (10, 3); the plain stand-off is (9.2, 3.0)
    brain.grid = room_grid(blocks=[(8.6, 2.4, 9.8, 3.6)])          # a cabinet right where the robot would stop
    pts = brain.approach_points(forklift, (2.0, 3.0), 0.8)
    assert pts and (9.2, 3.0) not in [(round(x, 1), round(y, 1)) for x, y in pts]
    assert all(brain.grid.is_free(x, y, brain.clearance) for x, y in pts)
    assert all(math.hypot(x - 10.0, y - 3.0) == pytest.approx(0.8, abs=1e-6) for x, y in pts)
    brain.grid = room_grid(unknown_above=3.2)                      # object at the edge of the explored area
    assert all(y < 3.2 for _, y in brain.approach_points(forklift, (2.0, 3.0), 0.8))       # only explored cells


def test_goal_is_not_sent_into_a_wall_when_the_map_is_known(env):
    world, brain = env
    brain.grid = room_grid(blocks=[(8.6, 2.4, 9.8, 3.6)])
    brain.execute(Command('GOAL', target='forklift'))
    assert wait_for(lambda: len(world.goals) == 1)
    g = world.goals[0].pose.position
    assert brain.grid.is_free(g.x, g.y, brain.clearance)
    assert math.hypot(g.x - 10.0, g.y - 3.0) == pytest.approx(0.8, abs=0.05)


def test_relative_move_is_shortened_when_the_target_is_blocked_and_refused_when_blocked_at_once(env):
    world, brain = env                                             # robot (2, 3) facing +y
    brain.grid = room_grid(unknown_above=5.0)                      # the rest of the way is unexplored
    msg = brain.execute(Command('RELATIVE', dx=4.0, dy=0.0))       # would end at y = 7, in unexplored space
    assert wait_for(lambda: len(world.goals) == 1)
    assert 4.0 < world.goals[0].pose.position.y < 5.0 and 'shortened' in msg
    brain.grid = room_grid(blocks=[(-5.0, 3.2, 15.0, 6.0)])        # solid wall 0.2 m in front, target inside it
    world.goals.clear()
    msg = brain.execute(Command('RELATIVE', dx=2.0, dy=0.0))
    time.sleep(0.3)
    assert world.goals == [] and 'Not moving' in msg
    brain.grid = room_grid(blocks=[(-5.0, 3.2, 15.0, 3.5)])        # thin wall, free floor behind it: Nav2 may go around
    brain.execute(Command('RELATIVE', dx=2.0, dy=0.0))
    assert wait_for(lambda: len(world.goals) == 1)


def test_a_group_is_approached_at_its_edge_not_its_centre(env):
    world, brain = env
    brain.graph.objects.append(SceneObject('desk_1', 'desk', 10.0, 3.0, 0.5, 9, 0.5, kind='group', members=4, size=4.0))
    try:
        brain.execute(Command('GOAL', target='desk_1'))
        assert wait_for(lambda: len(world.goals) == 1)
        assert world.goals[0].pose.position.x == pytest.approx(10.0 - (0.8 + 2.0), abs=0.05)
    finally:
        brain.graph.objects.pop()


def test_no_planning_when_the_robot_is_already_at_the_standoff(env):
    world, brain = env
    world.pose = (9.2, 3.0, 0.0)
    assert wait_for(lambda: brain.get_pose() and abs(brain.get_pose()[0] - 9.2) < 1e-6)
    brain.execute(Command('GOAL', target='forklift'))
    assert wait_for(lambda: len(world.goals) == 1)
    assert world.plan_queries == []


def test_a_failing_model_does_not_end_the_session(env):
    world, brain = env
    calls = []

    def llm(system_prompt, user_text, image):
        calls.append(user_text)
        if len(calls) == 1:
            raise RuntimeError('CUDA out of memory')
        return 'GOAL:forklift'

    lines, out = iter(['go to the forklift', 'go to the forklift', 'exit']), []
    chat_loop(brain, llm, input_fn=lambda _: next(lines), output_fn=out.append)
    assert any('[ERROR]' in o and 'CUDA out of memory' in o for o in out)
    assert wait_for(lambda: len(world.goals) == 1)                     # the second attempt worked


def test_wait_for_result_reports_the_outcome(env):
    world, brain = env
    lines, out = iter(['go to the forklift', 'exit']), []
    chat_loop(brain, lambda *a: 'GOAL:forklift', input_fn=lambda _: next(lines), output_fn=out.append,
              wait_for_result=True, result_timeout=10)
    assert '[NAV] Result: SUCCEEDED' in out


def test_stop_also_zeroes_the_wheels_directly(env):
    world, brain = env
    brain.stop_robot()
    assert wait_for(lambda: any(t.linear.x == 0.0 and t.angular.z == 0.0 for t in world.cmd_vels))


def test_roomy_stop_points_are_tried_before_merely_legal_ones(env):
    world, brain = env
    forklift = GRAPH.objects[0]                                     # (10, 3); the plain stand-off is (9.2, 3.0)
    brain.grid = room_grid(blocks=[(8.4, 3.6, 9.6, 3.7)])           # a wall 0.6 m north of the plain stand-off point
    pts = brain.approach_points(forklift, (2.0, 3.0), 0.8)
    assert pts and brain.grid.clearance(*pts[0]) >= brain.preferred_clearance
    assert any(brain.clearance <= brain.grid.clearance(*p) < brain.preferred_clearance for p in pts) \
        or all(brain.grid.clearance(*p) >= brain.preferred_clearance for p in pts)
    clearances = [brain.grid.clearance(*p) >= brain.preferred_clearance for p in pts]
    assert clearances == sorted(clearances, reverse=True)           # every roomy point comes before every tight one
