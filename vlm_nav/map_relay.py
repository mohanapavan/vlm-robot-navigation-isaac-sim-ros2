"""Relay RTAB-Map's /rtabmap/map onto /map as a latched (transient-local) topic.

Nav2's static costmap layer subscribes to /map with transient-local durability, so
without this relay there is no map for Nav2 to plan on (there is no map_server or
amcl in this camera-only setup).
"""
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy


class MapRelay(Node):
    def __init__(self, in_topic='/rtabmap/map', out_topic='/map'):
        super().__init__('map_relay')
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(OccupancyGrid, out_topic, latched_qos)
        # Default (volatile) QoS on the input matches whatever durability RTAB-Map publishes with.
        self.sub = self.create_subscription(OccupancyGrid, in_topic, self.cb, 10)

    def cb(self, msg):
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
