# Full Command Reference

Every command, in order, exactly as used. Each `T#` block runs in its own terminal.
The scene is [`scene/slam.usd`](../scene/slam.usd), opened and played in NVIDIA Isaac Sim.

---

## Stage 1 — SLAM mapping + data collection

**T1 — Static TF** (`base_link` → `base_scan`):
```bash
ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0.1 \
  --yaw 0 --pitch 0 --roll 0 \
  --frame-id base_link \
  --child-frame-id base_scan \
  --ros-args -p use_sim_time:=true
```

**T2 — SLAM Toolbox:**
```bash
ros2 run slam_toolbox async_slam_toolbox_node \
  --ros-args \
  -p use_sim_time:=true \
  -p odom_frame:=odom \
  -p map_frame:=map \
  -p base_frame:=base_link \
  -r scan:=/scan
```

**T3 — Teleop to drive around:**
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Drive around the entire environment, then **save the map**:
```bash
ros2 run nav2_map_server map_saver_cli -f ~/my_map
```

**Record a bag while mapping** (camera + lidar + odom + tf) — used later to build the scene graph:
```bash
ros2 bag record \
  /front_stereo_camera/left/image_raw \
  /front_3d_lidar/lidar_points \
  /chassis/odom \
  /tf \
  -o ~/warehouse_bag
```

---

## Stage 2 — Nav2 navigation on the saved map

**T1 — Static TF** (same as above).

**T2 — SLAM Toolbox** (keep running for the `map` frame — same as above).

**T3 — Nav2:**
```bash
ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=True \
  map:=/root/my_map.yaml
```

**Test goal:**
```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 1.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

---

## Stage 3 — Open-vocabulary scene graph (GroundingDINO)

Install (see [`setup/install_groundingdino.sh`](../setup/install_groundingdino.sh)):
```bash
mkdir -p ~/scene_graph
cd ~
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e .

mkdir -p ~/weights && cd ~/weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

Build the scene graph from the recorded bag:
```bash
python3 perception/build_scene_graph.py
# → writes ~/scene_graph/scene_graph.json  (object label → averaged x,y,z + count)
```

---

## Stage 4 — Natural-language robot brain (Qwen2.5-VL)

Set up the environment (see [`setup/install_qwen.sh`](../setup/install_qwen.sh)):
```bash
python3 -m venv ~/qwen_env
source ~/qwen_env/bin/activate
pip install torch transformers accelerate qwen-vl-utils rclpy
```

Run the live pipeline (Nav2 from Stage 2 must be running):
```bash
# T1 static TF, T2 slam_toolbox, T3 nav2  (as in Stage 2)

# T4 — Robot brain
source ~/qwen_env/bin/activate
python3 robot_brain/robot_brain.py
```

Then type natural-language commands, e.g. *"go to the forklift"*, *"go forward 1 meter"*,
*"what do you see?"*, *"stop"*. The brain replies with `GOAL:(x,y)`, `RELATIVE:(dx,dy)`,
`STOP`, or a description, and dispatches Nav2 goals accordingly.
