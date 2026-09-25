# Changelog

## 0.2.0

The repository is now a ROS 2 package (`vlm_nav`), the scene graph holds real object positions in the map frame,
and the SLAM + navigation pipeline starts from launch files. Numbers refer to the review list; "Test" names the
test that guards each fix (`tests/`).

### Core idea

| # | Problem | Fix | Test |
|---|---|---|---|
| 1 | The scene graph stored the **robot's** position when it saw an object; GroundingDINO boxes were never used | `build_scene_graph.py` back-projects each box centre with **stereo depth** (`stereo.py`, baseline from the right `camera_info`) into the camera optical frame and then into `map` | `test_build_scene_graph.py::test_object_is_located_in_the_map_frame` |
| 2 | Odom-frame positions used as `map`-frame Nav2 goals | Positions come from the bag's `/tf` (`map -> ... -> camera optical`) at the frame's stamp; the brain reads the robot pose from `map -> base_link` TF, not from odometry | same; `test_ros_brain.py::test_pose_comes_from_map_to_base_link_tf` |
| 3 | Relative moves ignored heading | `rotate_relative_offset(x, y, yaw, dx, dy)`; yaw from the TF quaternion | `test_geometry.py`, `test_ros_brain.py::test_relative_move_uses_robot_heading` |
| 4 | `STOP` matched anywhere in the reply, before `GOAL` | `GOAL`/`RELATIVE` are checked first; `STOP` only when the first non-empty line is a bare `STOP` | `test_command_parser.py` |

### Wrong behaviour

| # | Problem | Fix | Test |
|---|---|---|---|
| 5 | All same-class objects averaged into one point | Greedy clustering per label (`--cluster-radius`), merge of drifted clusters, `--min-observations` noise filter; ids `box_1`, `box_2`, ... | `test_scene_graph.py::test_same_class_objects_are_not_averaged_into_one` |
| 6 | The LLM copied coordinates from the prompt | The prompt lists object **names** only; `GOAL:<name>`; the code resolves id/label (bare label = nearest instance) and always applies the stand-off | `test_prompt_lists_names_but_never_object_coordinates`, `test_goal_by_bare_label_picks_nearest_instance` |
| 7 | Goal orientation always `w = 1.0` | Goal yaw points from the stand-off point at the object; relative moves keep the current heading | `test_goal_is_in_map_frame_with_standoff_and_faces_the_object` |
| 8 | Default (reliable) QoS on best-effort sensor topics | `qos_profile_sensor_data` for the camera subscription; odometry is no longer subscribed | `test_camera_frames_arrive_with_sensor_data_qos` |
| 9 | Image decoding assumed 3 channels | `image_utils.py` honours `encoding`, `step` and endianness (rgb8/bgr8/rgba8/bgra8/mono8/16-bit/32FC1) | `test_image_utils.py` |
| 10 | Missing GroundingDINO normalisation | `detector.py` uses GroundingDINO's own transform (resize + ImageNet mean/std) | *(needs the model)* |
| 11 | "Whatever odom came last" | Left/right images are paired by **header stamp** (`pairing.py`) and TF is looked up at that stamp; all TF is loaded before any image is processed, so bag order is irrelevant | `test_pairing.py`, `test_build_scene_graph.py` (bag written right-before-left, TF after images) |
| 12 | Raw phrases used as class names, scores dropped | `match_phrase_to_class` maps phrases back to the class list (merged/empty phrases dropped); scores are kept and averaged per object | `test_scene_graph.py::test_phrase_mapping` |

### Cleanups

| # | Fix |
|---|---|
| 13 | `_result_callback` reports `SUCCEEDED / ABORTED / CANCELED / REJECTED` (`last_result`, `result_event`, log) and ignores results of goals that were replaced |
| 14 | `wait_for_server(timeout_sec=--server-timeout)`; `send_goal` returns `False` and the console says Nav2 is unavailable |
| 15 | No hard-coded `/home/user/...`: `argparse` options with `~`-based defaults for every path and threshold (`--help`) |
| 16 | Latest camera frame and goal state are guarded by a lock |
| 17 | `package.xml`, `setup.py`, launch files (`slam.launch.py`, `navigation.launch.py`), `ros2 run vlm_nav ...` entry points |
| 18 | `requirements/{perception,brain,dev,constraints}.txt` with pinned versions; setup scripts use venvs with `--system-site-packages` (`rclpy` is not on PyPI) |

### Added: lidar SLAM pipeline

The original design used `slam_toolbox` on the lidar; the camera-only RTAB-Map pipeline is now an alternative.
`launch/slam_lidar.launch.py` starts `slam_toolbox` (`config/slam_toolbox_lidar.yaml`) plus the static transforms the
simulator does not provide (`front_2d_lidar -> base_scan`, the camera optical frames). `navigation.launch.py sensor:=lidar`
uses `config/nav2_params_lidar.yaml` (static map layers, `/chassis/odom`). On the running simulator: the map is
published immediately, the `map -> base_link / base_scan / camera_optical` TF chain resolved in 2061 of 2061 lookups, and
Nav2's planner returned paths without moving the robot.

### Added: found by running the real models (GroundingDINO + Qwen2.5-VL on the RTX 5080)

- **GroundingDINO runs on the GPU without `nvcc`**: the detector routes the missing compiled CUDA op to GroundingDINO's own
  pure-PyTorch implementation (~0.3 s/frame). Scene graph built from the recorded bag: 34 objects in the map frame;
  their median distance to the nearest lidar-mapped obstacle is 0.77 m (a random free cell: 2.25 m; forklifts 0.40 m).
- **Prompt rewritten and evaluated on the real model** (34 test commands): the inherited prompt answered "go forward 1 meter"
  with `GOAL:forklift_1`, parroted its instructions, and made "stop" a no-op. The new prompt (compact object list, worked
  examples generated from the real graph, direction-word `MOVE:left 1.5` grammar) scores 34/34.
- **STOP accepts `STOP - explanation`** (first word), not only a bare `STOP` line: the stricter rule from review item 4
  would have ignored the model's natural reply. Prose that merely contains "stop" still cannot trigger it.
- **Questions never move the robot**: messages starting with which/what/where/who/why/how use an answer-only prompt and
  are never acted on.
- **Goal selection asks Nav2's planner**: nearest reachable instance, at the stand-off and then progressively further
  out; refuses cleanly if nothing is reachable (the nearest forklift's plain approach point had no path in the real map).
- **Nav2 lidar pipeline: intermittent aborts and false "Reached the goal!" (root cause found).** With a `/scan`
  `ObstacleLayer` in a Nav2 costmap, that costmap's TF handling stopped after start-up. The *global* costmap (planner) froze
  its robot pose at the start-up position, so every plan began where the robot had been; from anywhere else the controller
  discarded the plan as out of range and aborted (`Resulting plan has 0 poses in it`), which is why goals only worked near
  the start-up position. The *local* costmap (controller) kept a single `map->odom` sample (`Transform data too old when
  converting from map to odom`) and Nav2 declared "Reached the goal!" instantly whenever the robot stood at the map origin.
  Found by comparing each costmap's published footprint with the true robot pose. The lidar parameter file now uses only the
  static `/map` layer in both costmaps (`tests/test_configs.py` guards it). Verified: a goal from 3.9 m away succeeds and
  the global costmap's timestamp tracks the sim clock. The camera pipeline never had an obstacle layer.
- **slam_toolbox in localization mode re-published a slightly different-sized map from time to time**, resetting Nav2's
  costmaps mid-drive. `slam_lidar.launch.py mode:=localization` now serves the saved map to Nav2 with `nav2_map_server`
  (`map_yaml:=`, default `~/my_map.yaml`) and remaps slam_toolbox's own map to `/slam_toolbox/map`.
- `slam_toolbox` TF publish period 0.1 s (10 Hz).
- `scripts/pipeline.sh start|stop|status`: SLAM (mapping or localization) then Nav2 in the required order.
- Per-class clustering radius; `record_bag` shutdown fix; `slam_lidar.launch.py mode:=localization` to resume from a saved
  pose graph; setup scripts fixed for the venv-sees-old-apt-packages problems (`setuptools`, `packaging`, `jinja2`).

### Found by running the pipeline (not in the review list)

- **Fixed: no `/clock`.** Neither `scene/slam.usd` nor the Nova Carter asset publishes it, so every `use_sim_time:=true`
  node waits forever. `scripts/add_clock_graph.py` adds the graph (the same one Isaac Sim's menu builds) and
  `scene/slam.usd` now contains it.
- **Fixed: unusable stereo occupancy grid.** With RTAB-Map's defaults the grid is ~90% unknown, almost no free space, and the
  robot's own cell is occupied, so Nav2 refuses every goal. `slam.launch.py` passes tuned `Grid/*` parameters
  (`GRID_ARGS`); free space went from 0.5% to ~34% on the test run.
- **Fixed: STOP racing a goal.** A `STOP` arriving before Nav2 accepted the goal was ignored. A goal generation counter
  now cancels goals that are still in flight (`test_stop_right_after_sending_a_goal_still_cancels_it`).
- **Fixed: unpaired stereo frames.** Best-effort image streams drop frames unevenly, so identical left/right stamps are
  rare. `record_bag` records only complete pairs and the builder pairs before sampling.
- **Fixed: crash on exit** of `robot_brain` (spin thread was still running inside `rclpy` at shutdown).
- **Fixed: rclpy logging** at two severities from one call site raised in the goal-result callback.
- `map_relay.py`: entry point, latched publisher, clean shutdown.
- Setup scripts no longer `pip install rclpy` (does not exist on PyPI).
- **Not changed:** the EKF also publishes `odom -> base_link`, which the simulator publishes too. They agree to
  0.02 mm in simulation; set `publish_tf: false` on a real robot.

### Verification status

Verified on this machine (Ubuntu 22.04, Isaac Sim 6.0, ROS 2 Humble, RTX 5080):
unit tests; ROS-level tests against a fake Nav2; a synthetic-bag test of the builder; the SLAM launch against the
running simulator (TF chain, `/map`, stereo depth vs. lidar and vs. odometry); Nav2 + `RobotBrain` on the live stack
(`scripts/nav_smoke_test.py`: 4/4 checks passed on one run; an earlier run stalled after the first goal on an
overloaded machine).

**Not run:** GroundingDINO inference and Qwen2.5-VL inference (no torch, weights or model downloads on the machine). Their
wrappers (`detector.py`, `QwenChat`) are reviewed but untested; the requirement pins exist on PyPI but were not installed.
