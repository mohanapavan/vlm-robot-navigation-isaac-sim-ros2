"""End-to-end test of the scene-graph builder on a synthetic rosbag2 with known geometry.

A textured 0.8 m 'box' sits at a known position in the MAP frame. The robot drives toward it while
map->odom is deliberately offset and rotated, so reading odometry instead of TF gives a visibly wrong answer.
The bag is written adversarially: right frame before left, and all TF after all images.
"""
import json
import math

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
pytest.importorskip('rosbag2_py')
pytest.importorskip('tf2_ros')

import rosbag2_py  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.serialization import serialize_message  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from vlm_nav import build_scene_graph  # noqa: E402
from vlm_nav.detector import Detection  # noqa: E402
from vlm_nav.geometry import transform_point, transform_to_matrix, yaw_to_quat  # noqa: E402

W, H, FX, FY, CX, CY, BASELINE = 640, 480, 500.0, 500.0, 320.0, 240.0, 0.15
OPTICAL_Q = (0.5, -0.5, 0.5, -0.5)                 # base_link -> camera *_rgb, as published by Isaac Sim
MAP_ODOM = (4.0, -3.0, 0.9)                        # x, y, yaw of map -> odom
CAM_XYZ = (0.1, 0.075, 0.346)
OBJ_SIZE = 0.8
N_FRAMES, DT, SPEED = 20, 0.2, 0.75                # 4 s, robot drives 3 m forward along odom +x
T0 = 100.0                                         # sim-time origin


def _mat(xyz, q):
    return transform_to_matrix(xyz, q)


T_MAP_ODOM = _mat((MAP_ODOM[0], MAP_ODOM[1], 0.0), yaw_to_quat(MAP_ODOM[2]))
T_BASE_CAM = _mat(CAM_XYZ, OPTICAL_Q)
# object 6.5 m ahead of the robot's start, along its heading, expressed in the map frame
OBJ_MAP = transform_point(T_MAP_ODOM, (6.5, 0.0, CAM_XYZ[2]))


def _stamp(msg_stamp, t):
    msg_stamp.sec, msg_stamp.nanosec = int(t), int(round((t - int(t)) * 1e9))


def _tf(parent, child, xyz, q, t):
    m = TransformStamped()
    _stamp(m.header.stamp, t)
    m.header.frame_id, m.child_frame_id = parent, child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in xyz)
    m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = (float(v) for v in q)
    return m


def _image(arr, t, frame):
    m = Image()
    _stamp(m.header.stamp, t)
    m.header.frame_id = frame
    m.height, m.width, m.encoding, m.step, m.is_bigendian = arr.shape[0], arr.shape[1], 'rgb8', arr.shape[1] * 3, 0
    m.data = arr.tobytes()
    return m


def _info(right):
    m = CameraInfo()
    m.height, m.width = H, W
    m.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    m.p = [FX, 0.0, CX, -FX * BASELINE if right else 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
    return m


def render_scene():
    """Yield (t, left, right, box_xyxy, base_in_odom_x) for each frame."""
    rng = np.random.default_rng(3)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W + 64), dtype=np.uint8), (3, 3), 0)
    obj_tex = cv2.GaussianBlur(rng.integers(0, 255, (64, 64), dtype=np.uint8), (3, 3), 0)
    d_bg = 4
    for i in range(N_FRAMES):
        t = T0 + i * DT
        base_x = SPEED * i * DT
        T_map_cam = T_MAP_ODOM @ _mat((base_x, 0.0, 0.0), (0, 0, 0, 1)) @ T_BASE_CAM
        p = np.linalg.inv(T_map_cam) @ np.array([*OBJ_MAP, 1.0])
        Z = p[2]
        u, v = FX * p[0] / Z + CX, FY * p[1] / Z + CY
        half = FX * OBJ_SIZE / Z / 2.0
        d = FX * BASELINE / Z
        u0, u1, v0, v1 = int(round(u - half)), int(round(u + half)), int(round(v - half)), int(round(v + half))
        left = bg[:, 32:32 + W].copy()
        right = bg[:, 32 + d_bg:32 + d_bg + W].copy()
        patch = cv2.resize(obj_tex, (u1 - u0, v1 - v0), interpolation=cv2.INTER_LINEAR)
        left[v0:v1, u0:u1] = patch
        du = int(round(d))                                   # integer shift keeps the ground truth exact
        right[v0:v1, u0 - du:u1 - du] = patch
        L, R = (np.repeat(x[:, :, None], 3, 2).copy() for x in (left, right))
        L[0, :4, :] = i                                     # frame index for the fake detector
        yield t, L, R, (float(u0), float(v0), float(u1), float(v1)), Z, du


class BoxDetector:
    """Stand-in for GroundingDINO: returns the true box of the synthetic object, looked up by frame index."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        return [Detection('box', 0.8, self.boxes[int(rgb[0, 0, 0])])]


def write_bag(path, with_optical_static=True, right_stamp_offset=0.0):
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    for name, typ in [('/tf', 'tf2_msgs/msg/TFMessage'), ('/tf_static', 'tf2_msgs/msg/TFMessage'),
                      ('/l/image', 'sensor_msgs/msg/Image'), ('/r/image', 'sensor_msgs/msg/Image'),
                      ('/l/info', 'sensor_msgs/msg/CameraInfo'), ('/r/info', 'sensor_msgs/msg/CameraInfo')]:
        writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format='cdr',
                                                     offered_qos_profiles=''))
    n = [0]

    def put(topic, msg):
        n[0] += 1
        writer.write(topic, serialize_message(msg), n[0])

    put('/l/info', _info(False))
    put('/r/info', _info(True))
    boxes = []
    for t, L, R, box, _, _ in render_scene():
        boxes.append(box)
        put('/r/image', _image(R, t + right_stamp_offset, 'cam_right_optical'))      # right BEFORE left
        put('/l/image', _image(L, t, 'cam_optical'))
    # all TF written after all images: bag order != time order
    static = [_tf('base_link', 'cam_rgb', CAM_XYZ, OPTICAL_Q, 0.0)]
    if with_optical_static:
        static.append(_tf('cam_rgb', 'cam_optical', (0, 0, 0), (0, 0, 0, 1), 0.0))
    msg = TFMessage()
    msg.transforms = static
    put('/tf_static', msg)
    for i in range(N_FRAMES * 3 + 6):
        t = T0 - 0.1 + i * DT / 3
        msg = TFMessage()
        msg.transforms = [
            _tf('map', 'odom', (MAP_ODOM[0], MAP_ODOM[1], 0.0), yaw_to_quat(MAP_ODOM[2]), t),
            _tf('odom', 'base_link', (SPEED * (t - T0), 0.0, 0.0), (0, 0, 0, 1), t)]
        put('/tf', msg)
    return boxes          # the writer is flushed and closed when this function's scope ends


ARGS = ['--left-topic', '/l/image', '--right-topic', '/r/image', '--left-info-topic', '/l/info',
        '--right-info-topic', '/r/info', '--sample-period', '0.4', '--min-observations', '2']


def run(tmp_path, boxes, extra=()):
    out = tmp_path / 'sg.json'
    det = BoxDetector(boxes)
    rc = build_scene_graph.main(['--bag', str(tmp_path / 'bag'), '--output', str(out), *ARGS, *extra], detector=det)
    return rc, out, det


def test_object_is_located_in_the_map_frame(tmp_path):
    boxes = write_bag(tmp_path / 'bag')
    rc, out, det = run(tmp_path, boxes)
    assert rc == 0
    data = json.loads(out.read_text())
    assert data['version'] == 2 and data['frame_id'] == 'map'
    (obj,) = data['objects']
    assert obj['id'] == 'box_1' and obj['label'] == 'box' and obj['count'] >= 5
    err = math.hypot(obj['x'] - OBJ_MAP[0], obj['y'] - OBJ_MAP[1])
    assert err < 0.3, f"object at ({obj['x']}, {obj['y']}), expected {OBJ_MAP[:2]}"
    assert obj['z'] == pytest.approx(OBJ_MAP[2], abs=0.3)
    # reading odometry / ignoring map->odom would land ~ (6.5, 0): clearly different from the map position
    assert math.hypot(6.5 - OBJ_MAP[0], 0.0 - OBJ_MAP[1]) > 2.0
    assert det.calls >= 5


def test_sampling_and_pairing_are_by_header_stamp(tmp_path):
    boxes = write_bag(tmp_path / 'bag')
    _, out, det = run(tmp_path, boxes, ['--sample-period', '1.0'])
    assert det.calls == 4                              # 4 s of frames, one every >= 1 s


def test_unpaired_stereo_yields_no_objects_and_a_failure_code(tmp_path):
    boxes = write_bag(tmp_path / 'bag', right_stamp_offset=0.05)       # right frames 50 ms late: not the same frame
    rc, out, det = run(tmp_path, boxes)
    assert rc == 1 and det.calls == 0
    assert json.loads(out.read_text())['objects'] == []


def test_missing_optical_static_tf_is_reported_and_camera_frame_override_fixes_it(tmp_path):
    boxes = write_bag(tmp_path / 'bag', with_optical_static=False)
    rc, _, det = run(tmp_path, boxes)
    assert rc == 1 and det.calls == 0                                  # no map <- cam_optical: frames skipped
    rc, out, det = run(tmp_path, boxes, ['--camera-frame', 'cam_rgb'])
    assert rc == 0
    (obj,) = json.loads(out.read_text())['objects']
    assert math.hypot(obj['x'] - OBJ_MAP[0], obj['y'] - OBJ_MAP[1]) < 0.3


def test_baseline_override_is_honoured(tmp_path):
    boxes = write_bag(tmp_path / 'bag')
    _, out, _ = run(tmp_path, boxes, ['--baseline', '0.15'])            # same as camera_info: unchanged result
    (obj,) = json.loads(out.read_text())['objects']
    assert math.hypot(obj['x'] - OBJ_MAP[0], obj['y'] - OBJ_MAP[1]) < 0.3
    _, out, _ = run(tmp_path, boxes, ['--baseline', '0.30'])            # wrong baseline -> everything twice as far
    good = [o for o in json.loads(out.read_text())['objects']
            if math.hypot(o['x'] - OBJ_MAP[0], o['y'] - OBJ_MAP[1]) < 0.5]
    assert good == []
