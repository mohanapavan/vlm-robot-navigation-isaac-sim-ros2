"""Build the semantic scene graph offline from a recorded rosbag2.

For every sampled left camera frame:
  1. pair it with the right frame by header stamp and compute stereo depth,
  2. run GroundingDINO,
  3. back-project each box centre with its depth into the camera optical frame,
  4. transform that point into the map frame with the bag's own /tf at the frame's stamp,
then cluster the per-frame observations into distinct object instances.

Record the bag with (see docs/commands.md):
  /tf /tf_static  left+right image_raw  left+right camera_info   (and --use-sim-time)
"""
import argparse
import logging
import os
import sys

from .geometry import backproject, transform_point, transform_to_matrix
from .image_utils import image_msg_to_rgb
from .pairing import StereoPairer, stamp_ns
from .scene_graph import DEFAULT_CLASS_RADIUS, DEFAULT_CLASSES, Observation, SceneGraph, cluster_observations
from .stereo import StereoDepth, baseline_from_projection, box_depth

log = logging.getLogger('vlm_nav.build_scene_graph')


# ----------------------------------------------------------------------------- bag access

def _storage_id(bag_path):
    import yaml
    meta = os.path.join(bag_path, 'metadata.yaml')
    if os.path.isfile(meta):
        with open(meta) as f:
            return yaml.safe_load(f)['rosbag2_bagfile_information'].get('storage_identifier', 'sqlite3')
    return 'sqlite3'


def iter_bag(bag_path, topics):
    """Yield (topic, deserialized message, bag_time_ns) for the requested topics."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=_storage_id(bag_path)),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        raise RuntimeError(f'topics not in bag {bag_path}: {missing}. Available: {sorted(types)}')
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    msg_types = {t: get_message(types[t]) for t in topics}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        yield topic, deserialize_message(data, msg_types[topic]), t_ns


def load_tf_and_info(bag_path, left_info_topic, right_info_topic, tf_cache_seconds=6 * 3600):
    """Pass 1: fill a tf2 buffer from /tf + /tf_static and grab one camera_info per camera.

    All TF is loaded before any image is looked at, so the order messages happen
    to be stored in the bag no longer matters.
    """
    import tf2_ros
    from rclpy.duration import Duration

    buffer = tf2_ros.Buffer(cache_time=Duration(seconds=tf_cache_seconds))
    left_info = right_info = None
    n_tf = 0
    for topic, msg, _ in iter_bag(bag_path, ['/tf', '/tf_static', left_info_topic, right_info_topic]):
        if topic in ('/tf', '/tf_static'):
            for t in msg.transforms:
                if topic == '/tf_static':
                    buffer.set_transform_static(t, 'bag')
                else:
                    buffer.set_transform(t, 'bag')
                n_tf += 1
        elif topic == left_info_topic and left_info is None:
            left_info = msg
        elif topic == right_info_topic and right_info is None:
            right_info = msg
    if left_info is None or right_info is None:
        raise RuntimeError('camera_info missing from the bag for the left and/or right camera')
    log.info('Loaded %d transforms', n_tf)
    return buffer, left_info, right_info


# ----------------------------------------------------------------------------- per-frame mapping

class SceneMapper:
    """Turns a stereo pair + detector + TF into map-frame Observations."""

    def __init__(self, detector, tf_buffer, left_info, right_info, map_frame='map',
                 baseline=None, stereo_scale=0.5, min_depth=0.3, max_depth=15.0, camera_frame=None):
        self.detector = detector
        self.camera_frame = camera_frame
        self.tf = tf_buffer
        self.map_frame = map_frame
        self.info = left_info
        baseline = baseline or baseline_from_projection(list(right_info.p))
        if not baseline:
            raise RuntimeError('stereo baseline unknown: right camera_info P[3] is 0. Pass --baseline <metres>.')
        self.baseline = baseline
        self.stereo_scale, self.min_depth, self.max_depth = stereo_scale, min_depth, max_depth
        self._stereo = None
        self.skipped = {'no_tf': 0, 'no_depth': 0}

    def _stereo_for(self, width):
        if self._stereo is None:
            fx = self.info.k[0] * width / self.info.width
            self._stereo = StereoDepth(fx, self.baseline, scale=self.stereo_scale,
                                       min_depth=self.min_depth, max_depth=self.max_depth)
            log.info('Stereo: fx=%.1f px, baseline=%.3f m', fx, self.baseline)
        return self._stereo

    def _camera_to_map(self, frame_id, stamp):
        from rclpy.duration import Duration
        from rclpy.time import Time
        t = self.tf.lookup_transform(self.map_frame, frame_id, Time(nanoseconds=stamp), Duration(seconds=0))
        tr, q = t.transform.translation, t.transform.rotation
        return transform_to_matrix((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))

    def process_pair(self, left_msg, right_msg):
        import tf2_ros
        left = image_msg_to_rgb(left_msg)
        right = image_msg_to_rgb(right_msg)
        stamp = stamp_ns(left_msg.header)
        frame = self.camera_frame or left_msg.header.frame_id
        try:
            cam_to_map = self._camera_to_map(frame, stamp)
        except tf2_ros.TransformException as e:
            self.skipped['no_tf'] += 1
            log.warning('No %s <- %s transform at stamp %.3f: %s', self.map_frame, frame, stamp * 1e-9, e)
            return []

        detections = self.detector.detect(left)
        if not detections:
            return []
        depth = self._stereo_for(left.shape[1]).compute(left, right)
        h, w = left.shape[:2]
        sx, sy = w / self.info.width, h / self.info.height
        fx, fy = self.info.k[0] * sx, self.info.k[4] * sy
        cx, cy = self.info.k[2] * sx, self.info.k[5] * sy

        observations = []
        for det in detections:
            d = box_depth(depth, det.box)
            if d is None:
                self.skipped['no_depth'] += 1
                continue
            u, v = (det.box[0] + det.box[2]) / 2.0, (det.box[1] + det.box[3]) / 2.0
            p_map = transform_point(cam_to_map, backproject(u, v, d, fx, fy, cx, cy))
            observations.append(Observation(det.label, det.score, float(p_map[0]), float(p_map[1]), float(p_map[2])))
            log.info('  %-10s score=%.2f depth=%.2fm -> map (%.2f, %.2f, %.2f)',
                     det.label, det.score, d, *p_map)
        return observations


def build_observations(bag_path, mapper, left_topic, right_topic, sample_period_s, pair_tolerance_s):
    """Pass 2: pair the stereo frames by header stamp, sample complete pairs, and map each one."""
    pairer = StereoPairer(int(pair_tolerance_s * 1e9))
    period_ns = int(sample_period_s * 1e9)
    last_sample = None
    observations, frames = [], 0
    for topic, msg, _ in iter_bag(bag_path, [left_topic, right_topic]):
        stamp = stamp_ns(msg.header)
        pair = pairer.add_left(stamp, msg) if topic == left_topic else pairer.add_right(stamp, msg)
        if pair is None:
            continue
        pair_stamp = stamp_ns(pair[0].header)
        if last_sample is not None and 0 <= pair_stamp - last_sample < period_ns:
            continue
        last_sample = pair_stamp
        frames += 1
        log.info('Frame %d (t=%.2fs)', frames, pair_stamp * 1e-9)
        observations.extend(mapper.process_pair(*pair))
    return observations, frames, pairer.unmatched


# ----------------------------------------------------------------------------- CLI

def parse_args(argv=None):
    home = os.path.expanduser('~')
    p = argparse.ArgumentParser(description='Build a map-frame semantic scene graph from a rosbag2.')
    p.add_argument('--bag', default=os.path.join(home, 'warehouse_bag'), help='rosbag2 directory')
    p.add_argument('--output', default=os.path.join(home, 'scene_graph', 'scene_graph.json'))
    p.add_argument('--weights', default=os.path.join(home, 'weights', 'groundingdino_swint_ogc.pth'))
    p.add_argument('--config', default=os.path.join(home, 'weights', 'GroundingDINO_SwinT_OGC.py'),
                   help='GroundingDINO model config (.py)')
    p.add_argument('--classes', nargs='+', default=DEFAULT_CLASSES, help='object classes to detect')
    p.add_argument('--box-threshold', type=float, default=0.30)
    p.add_argument('--text-threshold', type=float, default=0.25)
    p.add_argument('--left-topic', default='/front_stereo_camera/left/image_raw')
    p.add_argument('--right-topic', default='/front_stereo_camera/right/image_raw')
    p.add_argument('--left-info-topic', default='/front_stereo_camera/left/camera_info')
    p.add_argument('--right-info-topic', default='/front_stereo_camera/right/camera_info')
    p.add_argument('--map-frame', default='map')
    p.add_argument('--camera-frame', default=None,
                   help='optical frame to use instead of the image header frame_id (must follow the '
                        'x-right/y-down/z-forward optical convention)')
    p.add_argument('--sample-period', type=float, default=1.0, help='seconds (header time) between processed frames')
    p.add_argument('--pair-tolerance', type=float, default=0.02, help='max left/right header stamp gap in seconds')
    p.add_argument('--baseline', type=float, default=None,
                   help='stereo baseline in metres (default: from right camera_info P)')
    p.add_argument('--stereo-scale', type=float, default=0.5, help='downscale factor for stereo matching (speed)')
    p.add_argument('--min-depth', type=float, default=0.3)
    p.add_argument('--max-depth', type=float, default=15.0)
    p.add_argument('--cluster-radius', type=float, default=1.0, help='same-class detections closer than this are one object')
    p.add_argument('--class-radius', nargs='*', metavar='LABEL=METRES',
                   default=[f'{k}={v}' for k, v in DEFAULT_CLASS_RADIUS.items()],
                   help='per-class clustering radius overriding --cluster-radius (big objects need more); '
                        'pass with no values to disable')
    p.add_argument('--min-observations', type=int, default=2, help='drop objects seen fewer times than this')
    p.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: auto)")
    return p.parse_args(argv)


def main(argv=None, detector=None):
    """Entry point. `detector` (anything with .detect(rgb) -> [Detection]) can be injected for testing."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if detector is None:
        from .detector import GroundingDinoDetector
        detector = GroundingDinoDetector(args.config, args.weights, args.classes, args.box_threshold,
                                         args.text_threshold, args.device)
    log.info('Reading TF and camera calibration from %s...', args.bag)
    tf_buffer, left_info, right_info = load_tf_and_info(args.bag, args.left_info_topic, args.right_info_topic)
    mapper = SceneMapper(detector, tf_buffer, left_info, right_info, args.map_frame, args.baseline,
                         args.stereo_scale, args.min_depth, args.max_depth, args.camera_frame)

    log.info('Processing images...')
    observations, frames, unmatched = build_observations(
        args.bag, mapper, args.left_topic, args.right_topic, args.sample_period, args.pair_tolerance)

    class_radius = {k: float(v) for k, v in (item.split('=') for item in args.class_radius)}
    objects = cluster_observations(observations, args.cluster_radius, args.min_observations, class_radius)
    graph = SceneGraph(objects, args.map_frame)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    graph.save(args.output)

    log.info('\nProcessed %d stereo frames (%d unpaired, skipped: %s), %d observations -> %d objects',
             frames, unmatched, mapper.skipped, len(observations), len(objects))
    for o in objects:
        log.info('  %-14s (%.2f, %.2f, %.2f)  seen %d x, mean score %.2f', o.id, o.x, o.y, o.z, o.count, o.score)
    log.info('Wrote %s', args.output)
    if not objects:
        log.error('No objects found. Check that the bag has /tf with the %s frame, matching stereo stamps, '
                  'and lower --min-observations / --box-threshold if needed.', args.map_frame)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
