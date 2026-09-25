"""Add a ROS 2 /clock publisher to an Isaac Sim scene.

scene/slam.usd (and the Nova Carter asset it references) publish sensors, TF and odometry but no /clock.
Every node in the pipeline runs with use_sim_time:=true, so without /clock they all wait forever.
This builds the same graph as Isaac Sim's menu Tools > Robotics > ROS 2 OmniGraphs > Clock
(OnPlaybackTick -> ROS2PublishClock, timestamped by IsaacReadSimulationTime) and saves it in the scene.

Run with Isaac Sim's own Python (not the system python):

    ~/isaacsim_standalone/python.sh scripts/add_clock_graph.py scene/slam.usd            # in place
    ~/isaacsim_standalone/python.sh scripts/add_clock_graph.py scene/slam.usd out.usd    # to a new file

The scene is left unchanged if it already contains a ROS2PublishClock node.
"""
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('scene', help='input .usd/.usda/.usdc')
parser.add_argument('output', nargs='?', help='output path (default: overwrite the input)')
parser.add_argument('--graph-path', default='/World/ROS_Clock')
args = parser.parse_args()
scene = os.path.abspath(args.scene)
output = os.path.abspath(args.output) if args.output else scene

from isaacsim import SimulationApp  # noqa: E402  (must be created before any omni import)

app = SimulationApp({'headless': True})

import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension('isaacsim.ros2.bridge')
for _ in range(5):
    app.update()

ctx = omni.usd.get_context()
ctx.open_stage(scene)
while ctx.get_stage_loading_status()[2] > 0:
    app.update()
for _ in range(5):
    app.update()
stage = ctx.get_stage()

existing = [str(p.GetPath()) for p in stage.Traverse()
            if p.GetTypeName() == 'OmniGraphNode' and 'ROS2PublishClock' in str(p.GetAttribute('node:type').Get())]
if existing:
    print(f'Scene already publishes /clock via {existing}; nothing to do.')
else:
    keys = og.Controller.Keys
    og.Controller.edit(
        {'graph_path': args.graph_path, 'evaluator_name': 'execution'},
        {
            keys.CREATE_NODES: [
                ('OnPlaybackTick', 'omni.graph.action.OnPlaybackTick'),
                ('ReadSimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
                ('PublishClock', 'isaacsim.ros2.bridge.ROS2PublishClock'),
                ('Context', 'isaacsim.ros2.bridge.ROS2Context'),
            ],
            keys.CONNECT: [
                ('OnPlaybackTick.outputs:tick', 'PublishClock.inputs:execIn'),
                ('Context.outputs:context', 'PublishClock.inputs:context'),
                ('ReadSimTime.outputs:simulationTime', 'PublishClock.inputs:timeStamp'),
            ],
            keys.SET_VALUES: [('ReadSimTime.inputs:resetOnStop', True)],
        })
    # Export only this layer: the Nova Carter / warehouse references stay references instead of being flattened in.
    stage.GetRootLayer().Export(output)
    print(f'Added {args.graph_path} (publishes /clock) -> {output}')

app.close()
