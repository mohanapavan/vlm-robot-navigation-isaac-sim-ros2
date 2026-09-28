# Full Command Reference

Every command, in order. Each numbered block is its own terminal unless noted. Everything that used to be
five or six hand-typed terminals (two static TFs, the EKF, RTAB-Map, `map_relay.py`, RViz) is now one launch file.

Conventions: `~/ws` is the ROS workspace, and every ROS terminal starts with

```bash
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
```

---

## Quick start (everything already set up)

With a saved map (`~/my_map.yaml` + `.pgm`) and saved detections (`~/scene_graph/scene_graph.json`), nothing needs to be
recorded, mapped or detected again. The repository ships the hospital scene's results in `saved_state/hospital/`
(see its README):

```bash
# 0. Put the shipped map and detections where the defaults look for them (once)
cd ~/ws/src/vlm_nav && cp saved_state/hospital/my_map.yaml saved_state/hospital/my_map.pgm ~/
mkdir -p ~/scene_graph && cp saved_state/hospital/scene_graph*.json ~/scene_graph/
# 1. Isaac Sim: opens scene/slam.usd, waits for its streamed assets, presses Play (first run downloads ~1 GB, later runs ~30 s)
~/ws/src/vlm_nav/scripts/launch_isaac.sh
# 2. Saved map (identity map->odom, no pose graph needed) + Nav2, in the background, in the right order.
#    Repeat this step after every Stop/Play of the simulator (sim time restarts at 0).
~/ws/src/vlm_nav/scripts/pipeline.sh start
# 3. Talk to the robot (a raw scene_graph.json is consolidated on load: see section 5)
source /opt/ros/humble/setup.bash && source ~/ws/install/setup.bash && source ~/qwen_env/bin/activate
python -m vlm_nav.robot_brain --scene-graph ~/scene_graph/scene_graph.json --ros-args -p use_sim_time:=true
# 4. When finished:
~/ws/src/vlm_nav/scripts/pipeline.sh stop
```

`pipeline.sh start` picks the mode itself: `localization` when `~/my_map_posegraph.posegraph` exists, `static` when only
`~/my_map.yaml` exists, else `mapping` (`pipeline.sh start static|localization|mapping` to choose; `RVIZ=false` to skip RViz).
Logs go to `~/pipeline_logs/`.

Check the stack against the real simulator (each needs the stack above running; the last two also need Qwen):

```bash
python3 scripts/check_map_alignment.py                       # does the live lidar sit on the saved map?  (must say ALIGNED)
python3 scripts/nav_smoke_test.py                            # relative move / named goal / STOP / goal after STOP (no LLM)
~/qwen_env/bin/python scripts/live_command_test.py           # 5 typed commands through Qwen -> Nav2 -> the robot
~/qwen_env/bin/python scripts/destination_benchmark.py       # 5 easy + 5 medium + 5 hard destinations, full table
~/qwen_env/bin/python scripts/record_demo.py --command "what do you see?" --command "go to the wheelchair"   # video + GIF
```

The Word report is rebuilt from the stored results with `pip install -r requirements/report.txt && python3 scripts/build_report.py`.

---

## 0. One-time machine setup

<details>
<summary>Isaac Sim + ROS 2 Humble on Ubuntu 22.04 (the exact steps used on the dev machine, RTX 5080)</summary>

```bash
# NVIDIA userspace driver (no kernel module) and Isaac Sim
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
cd ~ && wget https://us.download.nvidia.com/XFree86/Linux-x86_64/580.105.08/NVIDIA-Linux-x86_64-580.105.08.run
chmod +x NVIDIA-Linux-x86_64-580.105.08.run
sudo ./NVIDIA-Linux-x86_64-580.105.08.run --no-kernel-module
unzip ~/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64.zip -d ~/isaacsim_standalone
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
cd ~/isaacsim_standalone && ./post_install.sh

# ROS 2 Humble
sudo apt-get update && sudo apt-get install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update && sudo apt install -y ros-humble-desktop
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && source ~/.bashrc
sudo apt update && sudo apt install -y ros-humble-teleop-twist-keyboard ros-humble-navigation2 \
    ros-humble-nav2-bringup ros-humble-rtabmap-ros ros-humble-robot-localization
```
</details>

Build this repository as a ROS 2 package:

```bash
sudo apt install -y python3-colcon-common-extensions
mkdir -p ~/ws/src && cd ~/ws/src
git clone https://github.com/mohanapavan/vlm-robot-navigation-isaac-sim-ros2.git vlm_nav
cd ~/ws && colcon build --symlink-install
source install/setup.bash
```

---

## 1. Launch Isaac Sim

```bash
scripts/launch_isaac.sh [scene.usd]        # default scene/slam.usd; log in ~/pipeline_logs/isaac.log
```

This starts Isaac Sim, opens the scene, waits until its (streamed) assets are in, and presses Play. The scene's low test
obstacles (`/World/TestObstacles`) are deleted from the loaded stage; the `.usd` file is never edited
(`REMOVE_PRIMS="" scripts/launch_isaac.sh` keeps them, `REMOVE_PRIMS=/World/A,/World/B` removes others). Close any Isaac Sim
that is already running first. Doing it by hand still works:

```bash
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
cd ~/isaacsim_standalone && ./isaac-sim.sh --/renderer/activeGpu=0
```

then open `scene/slam.usd` and press **Play**. **Stop then Play** returns the robot to the map origin but restarts sim time
at 0, so restart SLAM/Nav2 (`pipeline.sh stop && pipeline.sh start`) afterwards. `scene/slam.usd` now contains a
`/World/ROS_Clock` graph, so `/clock` is published. Check it before going further:

```bash
ros2 topic hz /clock
```

> **`/clock` silent?** Every node below runs with `use_sim_time:=true` and waits forever without it. Scenes
> saved before this fix (or your own) can be patched with Isaac Sim's own Python:
> ```bash
> ~/isaacsim_standalone/python.sh scripts/add_clock_graph.py path/to/scene.usd
> ```

The scene publishes (verified): `/front_stereo_camera/{left,right}/{image_raw,camera_info}` (`rgb8`,
1920x1200, best-effort), `/chassis/odom`, `/chassis/imu`, `/front_stereo_imu/imu`, `/scan`, `/tf`, and
subscribes to `/cmd_vel`. The stereo baseline in the right `camera_info` is 0.15 m.

---

## 2. One-time DDS reset (only when nothing else is running yet)

Never while other ROS 2 nodes are alive: deleting these out from under live processes breaks their ability
to talk to anything new.

```bash
rm -f /dev/shm/fastrtps_* /dev/shm/fastdds_* /dev/shm/sem.fastrtps_* /dev/shm/sem.fastdds_*
```

---

## 3. SLAM (one terminal, leave running): pick ONE pipeline

Never run both at once: they would both publish `map -> odom`.

### 3A. Lidar + slam_toolbox (recommended: fast, reliable map)

```bash
ros2 launch vlm_nav slam_lidar.launch.py
```

This starts, with `use_sim_time`:

| Node | Purpose |
|---|---|
| `static_transform_publisher` `front_2d_lidar -> base_scan` | the scan's `frame_id` is `base_scan` but the simulator only publishes TF for `front_2d_lidar` (same place, same orientation) |
| 2x `static_transform_publisher` `*_rgb -> *_optical` | image headers use `*_optical`; only `*_rgb` exists in TF. Needed by `build_scene_graph` |
| `slam_toolbox` (`config/slam_toolbox_lidar.yaml`) | publishes the latched `/map` and the `map -> odom` TF |
| `rviz2` | `config/map_view.rviz` (disable with `rviz:=false`) |

The simulator already publishes `odom -> base_link`, so there is no EKF and no `map_relay`. A map shows up in RViz
immediately (even before the robot moves) and grows as you drive. In `mode:=localization` an additional `map_server`
serves your saved map (`~/my_map.yaml`) to Nav2 as `/map`, so the costmaps never change size mid-drive.

In `mode:=static` (`ros2 launch vlm_nav slam_lidar.launch.py mode:=static`) there is no slam_toolbox at all: `map_server` serves
the saved map and an identity `map -> odom` is published. That is exact when the robot starts where the map was built and
odometry does not drift (true in Isaac Sim after Stop -> Play, and checked by `scripts/check_map_alignment.py`: 100 % of
the live lidar endpoints land within 0.15 m of the saved map). It needs only `~/my_map.yaml`, not the pose graph.
The map is served through a corrected copy of the yaml: `map_saver` writes `free_thresh: 0.25`, which makes `map_server`
read the grey "unexplored" pixels as *free* (Nav2 saw 0 % unknown instead of 67 %); your yaml is not touched.

### 3B. Stereo camera + RTAB-Map (no lidar)

```bash
ros2 launch vlm_nav slam.launch.py
ros2 launch vlm_nav slam.launch.py fresh_map:=true          # delete ~/rtabmap.db on start (-d)
```

| Node | Purpose |
|---|---|
| 2x `static_transform_publisher` | `front_stereo_camera_{left,right}_rgb -> ..._optical` (identity) |
| `ekf_node` (`config/ekf.yaml`) | fuses `/chassis/odom` + `/chassis/imu` into `/odometry/filtered` |
| RTAB-Map (`stereo:=true`, visual odometry off) | `map -> odom` TF and `/rtabmap/map`, from the stereo pair + filtered odometry |
| `map_relay` | republishes `/rtabmap/map` as a latched `/map` for Nav2 |
| `rviz2` | `config/map_view.rviz` (disable with `rviz:=false`) |

**RTAB-Map only creates a map once the robot has moved** (it adds a node every 10 cm / 0.1 rad). If RViz says
*No map received* while the robot is standing still, that is expected: drive with teleop. `fresh_map` is false by
default: an existing database is continued, never deleted.

The RTAB-Map occupancy grid is tuned for a low, forward-looking stereo camera on a flat floor (`GRID_ARGS` in
`launch/slam.launch.py`). With RTAB-Map's defaults the stereo-only grid is ~90% unknown and marks the robot's own
cell occupied, so Nav2 can never plan. The camera map is still noisier than the lidar map.

Sanity checks (either pipeline):

```bash
ros2 run tf2_ros tf2_echo map base_link          # the map -> odom -> base_link chain is up
ros2 topic echo /map --once --field info         # a map exists (use this to pick goals inside it)
```

---

## 4. Drive around: build the map AND record the bag in the same run

Keep **Isaac Sim and SLAM running** for the rest of the session (see "Why the map frame matters" below). In two more
terminals:

```bash
# terminal A: drive (slowly, pass within 2-5 m of the objects, look at every object from a couple of sides)
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

```bash
# terminal B: record one complete stereo pair per second + all TF (start it before you drive, Ctrl+C at the end)
ros2 run vlm_nav record_bag --output ~/warehouse_bag --ros-args -p use_sim_time:=true
```

The bag holds: left+right `image_raw`, both `camera_info`, `/tf`, `/tf_static`. A full-rate 1920x1200 stereo bag would
be hundreds of MB/s, so `record_bag` keeps one pair per second of header time (`--period`). **Start SLAM first**: the
bag must contain the `map -> odom` transform, which only exists while SLAM is running.

When you are done driving, save a snapshot of the map (reference only):

```bash
ros2 run nav2_map_server map_saver_cli -f ~/my_map          # 3A (slam_toolbox publishes /map)
ros2 run nav2_map_server map_saver_cli -f ~/my_map_visual --ros-args -r map:=/rtabmap/map    # 3B
```

**Save the SLAM state so you can resume later** (lidar pipeline; do this after driving):

```bash
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '/home/user/my_map_posegraph'}"
```

Later (a new day, a restarted simulator), restore the same map instead of mapping again: restart Isaac Sim (Stop, then
Play: the robot returns to its start pose, which is the map origin) and launch SLAM in localization mode:

```bash
ros2 launch vlm_nav slam_lidar.launch.py mode:=localization
```

`map_file:=` selects another saved pose graph (default `~/my_map_posegraph`, no extension). The `map` frame is restored,
so the scene-graph coordinates stay valid.

**Why the map frame matters.** The scene graph stores object positions in the `map` frame of the SLAM run that was
active during recording. Nav2 and the brain must use that same `map` frame. So do **not** restart SLAM (or the
simulator) between recording and navigating: build the scene graph while everything is still running, then start
Nav2 and the brain. (A fresh SLAM run puts `map` at the robot's new start pose, which shifts every object.)

---

## 5. Build the semantic scene graph (GroundingDINO)

Uses its own venv (GroundingDINO needs an older `transformers` than Qwen). The venv is created with
`--system-site-packages` because `rclpy`, `tf2_ros`, `rosbag2_py` and `cv2` come from the ROS install:

```bash
source /opt/ros/humble/setup.bash
bash setup/install_groundingdino.sh          # -> ~/perception_env, ~/weights
```

```bash
source /opt/ros/humble/setup.bash && source ~/ws/install/setup.bash
source ~/perception_env/bin/activate
python -m vlm_nav.build_scene_graph --bag ~/warehouse_bag --output ~/scene_graph/scene_graph.json
```

(`python -m ...` inside the venv, not `ros2 run`: `ros2 run` starts the system Python, which has no torch.)
On the GPU it takes ~0.3 s per frame. The machine has no `nvcc`, so GroundingDINO's compiled CUDA op is missing; the
detector automatically uses GroundingDINO's pure-PyTorch version of the same operation, still on the GPU.

For each sampled stereo pair it runs stereo matching for metric depth, detects objects, back-projects each
box centre into the camera optical frame, transforms it into `map` with the bag's own `/tf` at the frame's
timestamp, and clusters the sightings into distinct instances (`box_1`, `box_2`, `forklift_1`, ...).
All paths and thresholds are arguments (`--help`): `--classes`, `--box-threshold`, `--sample-period`,
`--cluster-radius`, `--min-observations`, `--baseline`, `--camera-frame`, ...

Output (`scene_graph.json`, version 2):

```json
{"version": 2, "frame_id": "map",
 "objects": [{"id": "forklift_1", "label": "forklift", "x": 4.1, "y": -2.3, "z": 0.5, "count": 6, "score": 0.62}]}
```

### From raw detections to a small, trustworthy graph

`build_scene_graph` now finishes with `consolidate.py`, and `robot_brain` applies it to any raw graph it loads
(`--no-consolidate` to opt out). You can also run it on a saved file, without the bag or the detector:

```bash
python -m vlm_nav.consolidate --input ~/scene_graph/scene_graph.json --map ~/my_map.yaml \
    --output ~/scene_graph/scene_graph.consolidated.json
```

What it does, and why (see CHANGELOG 0.3.0 for the measurements): walls are structure, not destinations; duplicates of one
object are merged; two class names on one spot are resolved by evidence; a pile of identical things (20 stacked boxes)
becomes ONE entry `{"kind": "group", "members": N, "size": metres, "z_min", "z_max"}` instead of `box_1 ... box_20`;
each entry gets an importance (class weight x evidence x plausibility on the saved map) and weak ones are dropped, though
every detected class keeps its best entry. `--min-importance`, `--max-objects`, `--conflict-radius`, `--no-groups` tune it;
the per-class radii live in `CLASS_PROFILES` (unknown classes get sensible defaults).

Check any graph against the simulator's real objects (`evaluation/hospital_ground_truth.json`):

```bash
python3 scripts/eval_scene_graph.py ~/scene_graph/scene_graph.json ~/scene_graph/scene_graph.consolidated.json
```

For a new recording, `build_scene_graph` also suppresses duplicate boxes per frame, counts distinct *views* (not boxes),
ignores detections farther than `--max-object-range 8` m, saves `*.observations.json` and `*.raw.json` next to the output, and
`--from-observations FILE` redoes the clustering + consolidation later without the bag. `--raw` skips consolidation.

Positions are those of the **object surface facing the camera**, in the SLAM `map` frame. Big objects are clustered with a larger
radius (`--class-radius forklift=2.0 wall=4.0 ...`) so one forklift is not reported as several instances.

---

## 6. Navigation (one terminal, leave running)

```bash
ros2 launch vlm_nav navigation.launch.py sensor:=lidar     # after 3A
ros2 launch vlm_nav navigation.launch.py                   # after 3B (camera)
```

`sensor:=lidar` uses `config/nav2_params_lidar.yaml` (static map layers only, reads `/chassis/odom`);
the default uses `config/nav2_params_camera.yaml`.

Wait for `Managed nodes are active`, then test with a goal inside the mapped area:

```bash
ros2 topic echo /map --once --field info        # check bounds first
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 1.0, z: 0.0}, orientation: {w: 1.0}}}}" --feedback
```

Goals that fall inside walls, inflated obstacles or outside the explored map are refused by the planner and
the action ends `ABORTED`.

---

## 7. Talk to the robot (Qwen2.5-VL)

```bash
source /opt/ros/humble/setup.bash
bash setup/install_qwen.sh                    # -> ~/qwen_env
```

```bash
source /opt/ros/humble/setup.bash && source ~/ws/install/setup.bash
source ~/qwen_env/bin/activate
python -m vlm_nav.robot_brain --scene-graph ~/scene_graph/scene_graph.json --ros-args -p use_sim_time:=true
```

| You say | Model replies | Robot does |
|---|---|---|
| "go to the forklift" | `GOAL:forklift_1` (or `GOAL:forklift`) | Nav2 goal near the nearest **reachable** forklift, facing it |
| "go to forklift_3" | `GOAL:forklift_3` | same, for that instance |
| "go forward 1 meter" | `MOVE:forward 1` | Nav2 goal 1 m ahead of the robot's current heading |
| "move left 2 meters" | `MOVE:left 2` | 2 m to the robot's left (`back`, `right` likewise) |
| "stop" | `STOP` | cancels the active goal |
| "what do you see?" / "which forklift is closest?" | plain text | answers in words; **never moves** |
| "go to the spaceship" | plain text | says it doesn't know that object |

Messages that start with a question word (which / what / where / who / why / how) get an answer-only prompt and are
never acted on, even if the model replied with a command. `RELATIVE:(dx,dy)` is still accepted as an alternative to
`MOVE:`.

Names are forgiving: `vending_machine_2`, `the trash cans`, `beds` and `Trash Can 2` all resolve. If you name only a class
("take me to the cart") the nearest instance is used even when the model answered with an arbitrary `cart_1`; if you name an
instance ("go to cart_3") that instance wins over a different one the model wrote. A **group** ("group of 5, 7 m wide") is
approached at its edge (stand-off + its radius), not its centre. With `/map` available, approach points are picked on known
free cells with wall clearance, and a `MOVE` whose end point is inside a wall or unexplored space is shortened to the last free
point (or refused with a message). `STOP` cancels the goal *and* publishes a zero `/cmd_vel`. A failing model call is reported
and the session continues; `--wait-result` (automatic when stdin is not a terminal) waits for each goal and prints
`[NAV] Result: ...`.

**Choosing where to stop.** Before sending a `GOAL`, the brain asks Nav2's planner whether the approach point is
reachable (`--max-plan-checks`). It tries the nearest instance first, at the stand-off distance and then 0.4 / 0.8 /
1.6 m further out (an approach point 0.8 m from a surface can fall inside the obstacle's inflated zone). If nothing is
reachable it says so and does not move.

The console prints `[NAV] ...` for what was sent, and the node logs `Navigation finished: SUCCEEDED / ABORTED /
CANCELED` when Nav2 reports back. Options: `--standoff`, `--model`, `--map-frame`, `--server-timeout`.

---

## 8. Which terminals to keep

- **Keep running, always:** Isaac Sim, the SLAM launch (`slam_lidar.launch.py` or `slam.launch.py`), `navigation.launch.py`. These provide `/map` and the live
  `map -> odom -> base_link` chain Nav2 depends on; if any dies, navigation has nothing to work with.
- **Stop when done:** teleop, `record_bag`, `build_scene_graph`. RViz can be closed and reopened freely.

---

## Tests

```bash
source /opt/ros/humble/setup.bash
python3 -m pytest tests -q
```

About 240 tests, flake8 clean (`python3 -m flake8 vlm_nav scripts tests launch`). Pure-logic tests run anywhere. The ROS tests use a fake Nav2 and fake TF on their own DDS domain (87), so they
never interfere with a running simulator.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| RViz: *No map received* | 3A: SLAM not running (or a different `/map` QoS in your own RViz config: use `config/map_view.rviz`). 3B: the robot has not moved yet: RTAB-Map only publishes a map after the first motion |
| Nothing moves, SLAM/Nav2 idle | `/clock` not published: play the scene, or patch it with `scripts/add_clock_graph.py` |
| Nav2: `failed to create plan` for every goal | Goal is inside/behind an obstacle or unexplored; or `/map` is empty. Check `ros2 topic echo /map --once --field info` and drive further |
| No camera images in a node | Isaac Sim publishes best-effort; subscribe with `qos_profile_sensor_data` |
| `build_scene_graph`: `No map <- ..._optical transform` | `/tf_static` was not recorded (the optical static TF lives there). Re-record, or `--camera-frame front_stereo_camera_left_rgb` |
| `build_scene_graph`: `stereo baseline unknown` | Right `camera_info` has `P[3] = 0`; pass `--baseline 0.15` |
| Objects at wrong places after loop closure | The bag stores the TF at record time; map corrections made later are not applied retroactively |
| Nav2 says `Reached the goal!` instantly and the robot does not move | Same cause as the next row (the goal cannot be transformed, so it defaults to (0, 0)). Never put an `ObstacleLayer` on `/scan` in either costmap of `nav2_params_lidar.yaml` |
| Nav2: `Transform data too old when converting from map to odom`, goals abort | The `map -> odom` transform's *timestamp* is not advancing: SLAM's own clock froze. Check `ros2 topic echo /map --field header.stamp` against `ros2 topic echo /clock`: they must move together. Restart the SLAM launch |
| Nav2 bring-up: `failed to send response to .../change_state (timeout)` | DDS shared-memory trouble. Stop all ROS nodes, then run the reset in section 2 and start again. Do not mix `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` processes with default ones (SLAM's clock froze while I did) |
| `ros2 run vlm_nav robot_brain` fails with `No module named torch` | Use `python -m vlm_nav.robot_brain` inside the venv (see section 7) |
| Robot ends up wedged against a forklift's forks | The forks sit below the 2-D lidar's scan plane, so the map does not show them. Drive out with a short forward `cmd_vel`, or press Stop then Play in Isaac Sim |
| Robot stuck against an object in the simulator | Press Stop then Play in Isaac Sim (robot returns to start), then restart SLAM (`mode:=localization` to keep the map) |
| TF jitter on `odom -> base_link` | The sim publishes it and the EKF republishes it. They agree to 0.02 mm here; on a real robot set `publish_tf: false` in `config/ekf.yaml` |
| Everything stalls / TF errors after pressing Stop then Play in Isaac Sim | Sim time restarted at 0 while SLAM/Nav2 kept the old clock: `scripts/pipeline.sh stop && scripts/pipeline.sh start` |
| Goals into "unexplored" parts of the map succeed | The saved map was served with `free_thresh: 0.25` (unknown read as free). `mode:=static` / `localization` now serve a corrected copy: `ros2 topic echo /map --once --field data` must contain `-1` |
| `check_map_alignment.py` says MISALIGNED | The simulator was not reset (Stop -> Play) or the map belongs to another start pose |
| Qwen loads with "some parameters offloaded to the cpu" | Isaac Sim and the 3B model share one 16 GB GPU; it still works, only slower. Close other GPU programs, or run Isaac with `--no-window` |
| `ModuleNotFoundError: numpy` ABI error | ROS 2 Humble needs NumPy 1.x; keep the pins in `requirements/constraints.txt` |
