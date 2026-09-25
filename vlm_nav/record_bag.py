"""Record a compact rosbag2 for build_scene_graph: throttled stereo images + camera_info + TF (+ optional /scan).

`ros2 bag record` of full-rate 1920x1200 stereo is hundreds of MB/s and the scene-graph builder only
looks at about one frame per second, so this keeps one complete stereo pair per --period seconds of *header* time
(sim time in Isaac Sim) and every /tf and /tf_static message.

    ros2 run vlm_nav record_bag --output ~/warehouse_bag --ros-args -p use_sim_time:=true
    # drive around with teleop, Ctrl+C to finish
"""
import argparse
import os
import sys

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_msgs.msg import TFMessage

from .pairing import StereoPairer, stamp_ns


class BagRecorder(Node):
    def __init__(self, output, left, right, left_info, right_info, period, scan_topic=None,
                 pair_tolerance=0.005, queue_depth=20):
        super().__init__('scene_bag_recorder')
        self.period_ns = int(period * 1e9)
        self._left, self._right = left, right
        self._pairer = StereoPairer(int(pair_tolerance * 1e9))
        self._last_bucket = None
        self._info_written = set()
        self.counts = {}
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(rosbag2_py.StorageOptions(uri=output, storage_id='sqlite3'),
                         rosbag2_py.ConverterOptions('cdr', 'cdr'))

        latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        # Isaac Sim publishes best-effort; a deeper queue rides out bursts of multi-MB frames.
        images = QoSProfile(depth=queue_depth, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._topic(left, 'sensor_msgs/msg/Image', Image, images, self._on_left)
        self._topic(right, 'sensor_msgs/msg/Image', Image, images, self._on_right)
        self._topic(left_info, 'sensor_msgs/msg/CameraInfo', CameraInfo, qos_profile_sensor_data,
                    self._on_info(left_info))
        self._topic(right_info, 'sensor_msgs/msg/CameraInfo', CameraInfo, qos_profile_sensor_data,
                    self._on_info(right_info))
        self._topic('/tf', 'tf2_msgs/msg/TFMessage', TFMessage, 100, self._plain('/tf'))
        self._topic('/tf_static', 'tf2_msgs/msg/TFMessage', TFMessage, latched, self._plain('/tf_static'))
        if scan_topic:
            self._topic(scan_topic, 'sensor_msgs/msg/LaserScan', LaserScan, qos_profile_sensor_data,
                        self._on_scan(scan_topic))

    def _topic(self, name, type_name, msg_type, qos, cb):
        self.writer.create_topic(rosbag2_py.TopicMetadata(
            name=name, type=type_name, serialization_format='cdr', offered_qos_profiles=''))
        self.create_subscription(msg_type, name, cb, qos)
        self.counts[name] = 0

    def _write(self, topic, msg):
        self.writer.write(topic, serialize_message(msg), self.get_clock().now().nanoseconds)
        self.counts[topic] += 1

    def _bucket(self, msg):
        return stamp_ns(msg.header) // self.period_ns

    def _on_left(self, msg):
        self._maybe_write(self._pairer.add_left(stamp_ns(msg.header), msg))

    def _on_right(self, msg):
        self._maybe_write(self._pairer.add_right(stamp_ns(msg.header), msg))

    def _maybe_write(self, pair):
        """Only complete stereo pairs are recorded, at most one per period."""
        if pair is None:
            return
        left, right = pair
        bucket = self._bucket(left)
        if bucket == self._last_bucket:
            return
        self._last_bucket = bucket
        self._write(self._left, left)
        self._write(self._right, right)

    def _on_info(self, topic):
        def cb(msg):
            if topic not in self._info_written:      # calibration never changes; one message is enough
                self._info_written.add(topic)
                self._write(topic, msg)
        return cb

    def _plain(self, topic):
        return lambda msg: self._write(topic, msg)

    def _on_scan(self, topic):
        return lambda msg: self._write(topic, msg) if self._bucket(msg) == self._last_bucket else None


def parse_args(argv):
    p = argparse.ArgumentParser(description='Record a throttled stereo + TF bag for build_scene_graph.')
    p.add_argument('--output', default=os.path.join(os.path.expanduser('~'), 'warehouse_bag'))
    p.add_argument('--left-topic', default='/front_stereo_camera/left/image_raw')
    p.add_argument('--right-topic', default='/front_stereo_camera/right/image_raw')
    p.add_argument('--left-info-topic', default='/front_stereo_camera/left/camera_info')
    p.add_argument('--right-info-topic', default='/front_stereo_camera/right/camera_info')
    p.add_argument('--period', type=float, default=1.0, help='seconds (header time) between recorded stereo pairs')
    p.add_argument('--scan-topic', default=None, help='also record this LaserScan topic (for verification)')
    p.add_argument('--pair-tolerance', type=float, default=0.005,
                   help='max left/right header stamp gap (s) to count as one stereo pair')
    return p.parse_args(argv)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    rclpy.init(args=argv)
    args = parse_args(rclpy.utilities.remove_ros_args(argv)[1:])
    node = BagRecorder(args.output, args.left_topic, args.right_topic, args.left_info_topic,
                       args.right_info_topic, args.period, args.scan_topic, args.pair_tolerance)
    node.get_logger().info(f'Recording to {args.output} (Ctrl+C to stop)')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # Ctrl+C makes rclpy shut its context down while a multi-MB image is being taken from the queue, which
        # surfaces as a RuntimeError. That is a normal stop; anything else is a real error.
        if rclpy.ok():
            raise
    finally:
        print(f'Recorded: {node.counts}')     # (the logger is already shut down by rclpy's SIGINT handler)
        del node.writer          # flush + write metadata.yaml
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
