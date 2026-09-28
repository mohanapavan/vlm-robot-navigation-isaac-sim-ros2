# Changelog

## 0.3.0

Tested on the live stack (Isaac Sim 6.0 hospital scene, saved lidar map, saved detections, Nav2, Qwen2.5-VL-3B on an
RTX 5080), then two rounds of changes: robustness fixes that keep the approach as it was, and a redesign of how detections
become a scene graph. Nothing was re-recorded: the saved map and `scene_graph.json` were used as they are.

### Robustness fixes (same approach: name -> lookup -> stand-off goal -> planner check)

| # | Problem found | Fix | Test |
|---|---|---|---|
| 19 | **Saved map served with no unexplored space.** map_saver writes `free_thresh: 0.25`; map_server then reads the grey "unknown" pixels (0.196) as *free*: Nav2's `/map` had 0 unknown cells instead of 67 % | `slam_lidar.launch.py` hands map_server a corrected copy of the yaml (yours is untouched); `MapGrid` reads it the same way | `test_map_grid.py` |
| 20 | No way to use a saved map without the slam_toolbox pose graph (`serialize_map` can deadlock) | `slam_lidar.launch.py mode:=static`: map_server + identity `map -> odom`; `scripts/check_map_alignment.py` confirms the live scan sits on the map (100 % of endpoints within 0.15 m) | live |
| 21 | Multi-word classes (`vending machine`, `trash can`): ids read `vending machine_3`, the model writes `vending_machine_3`, `the trash cans`, `beds` and gets "Unknown object" | one normalisation (case, `_`/`-`/space, articles, plurals) in the parser and in `SceneGraph.resolve` | `test_scene_graph.py`, `test_command_parser.py` |
| 22 | Goals could be placed inside walls or unexplored space; the planner then aborted | with `/map`, approach points are chosen on known free cells with wall clearance and a clear line to the object (ring around it, nearest to the robot first); relative moves whose target is blocked are shortened to the last free point, or refused with a message | `test_ros_brain.py` |
| 23 | `GOAL` when the robot already stood at the stand-off point was treated as "no path" (a 1-pose path) | already-there goals skip planning; a 1-pose path counts | `test_no_planning_when_...` |
| 24 | A failing model call (CUDA out of memory) or Nav2 answer ended the chat session | errors are reported and the loop continues; a dead goal request reports `ERROR` | `test_a_failing_model_...` |
| 25 | `STOP` relied only on the Nav2 cancel chain (slow at low real-time factor, useless if Nav2 is wedged) | `STOP` also publishes a zero `/cmd_vel`; drift after STOP on the live stack fell from 0.109 m to 0.096 m | `test_stop_also_zeroes_the_wheels_directly` |
| 26 | `needs_vision` matched substrings ("seat", "interview" contain "see"/"view") | whole words | `test_vision_triggers_are_whole_words` |
| 27 | Scripted runs left before a goal finished (the exit cancels it) | `--wait-result` (automatic when stdin is not a terminal) prints `Result: SUCCEEDED/...` | `test_wait_for_result_...` |
| 28 | A malformed entry made the whole scene graph unloadable | skipped and counted; unknown keys ignored; v2 files still load | `test_scene_graph.py` |
| 34 | A 3B model answered "take me to the cart" with an arbitrary `cart_1` (26 m away, the nearest was 4.4 m), and could write a different id than the one the user typed | `reconcile_goal`: a bare class means the nearest instance; an explicit id beats a different one the model wrote | `test_goal_agrees_with_what_the_user_said` |
| 35 | The prompt lists at most 6 objects per class, so `door_13` was hidden and the model said "I don't know an object called door_13" | objects the user names are always listed | `test_an_object_the_user_names_is_always_listed...` |
| 36 | Two consecutive destinations ABORTED after 1.4 m (`collision ahead`, `Failed to make progress`): the plan hugged a chair whose low base the 2-D lidar cannot see, and SimpleProgressChecker ignores turning | Nav2 lidar costmaps inflation 0.5 -> 0.9 m, scaling 3.0 -> 2.5 (plan 0.3 m longer, at least 0.77 m clear instead of 0.51); `PoseProgressChecker` (turning counts); stop points with 0.7 m clearance are tried first (`--preferred-clearance`) | `test_configs.py`, `test_roomy_stop_points_...` |
| 37 | "what do you see?" once answered with the robot's own pose as if it were the position of a seen object | the question prompt says the pose is the robot's, and to describe the picture | `test_question_prompt_says_the_pose_is_the_robots_...`, live |
| 29 | Manual "open the scene, press Stop, press Play" | `scripts/launch_isaac.sh` opens the scene, waits for its streamed assets and presses Play; `REMOVE_PRIMS` deletes prims (default `/World/TestObstacles`) from the loaded stage without editing the file | live |

### Perception: why 149 objects, and what changed

Measured against the simulator's own objects (`evaluation/hospital_ground_truth.json`, world frame = map frame, checked by
overlaying the saved map on the scene's wall geometry: best shift 0.0 m / 0 deg). Scored with `scripts/eval_scene_graph.py`.

| | entries | precision | recall |
|---|---|---|---|
| saved detections, walls excluded | 125 | 0.38 | 0.22 |
| consolidated (default settings) | 46 | 0.61 | 0.18 |

Findings on the raw graph:

* Detection score does not separate real from look-alike: the one true vending machine cluster scored like the 17 false ones
  (real vending machines: 2; entries: 18). What does help: evidence, agreement with nearby detections, and the map (an entry
  1 m or more from every mapped obstacle was 85 % wrong). An entry in unexplored space was real 38 % of the time (26
  entries), the same as average, so that alone is weak evidence: the plausibility penalty for it is a judgement call, and
  removing it changes the result only slightly (56 instead of 46 entries, precision 0.57 instead of 0.61).
* 54 of the 125 non-wall entries had a different-class entry within 1 m (67 of 149 if walls count): one object, several class
  names.
* 45 of 149 had exactly the minimum (2) observations; the count was *boxes*, and GroundingDINO returns several overlapping boxes
  per object without suppression, so one frame could make "seen twice".
* 24 entries were walls: structure, not a destination.
* Repeated things came out as `bed_1 ... bed_14`, with no notion of "many beds here".
* Recall is limited by the saved detections themselves (the raw graph covers 34 of 156 real objects), not by consolidation.

Changes:

| # | Fix | Test |
|---|---|---|
| 30 | `consolidate.py`: structure dropped; duplicates merged (per-class radius, size-capped); different-class conflicts on one spot resolved by evidence; piles collapsed into one **group** entry (`kind: group`, `members`, `size`, `z_min/z_max`, never wider than 6 m); importance = class weight x evidence x plausibility (height, map state, distance to mapped obstacles); low-importance entries dropped, each detected class keeps its best guess; ids are `trash_can_1`. Deterministic for any input order | `test_consolidate.py` (incl. 20 stacked boxes -> 1 entry) |
| 31 | Scene graph v3 (`kind`, `members`, `size`, `z_min`, `z_max`, `importance`, `consolidated`); v2 still loads; the brain consolidates a raw graph on load (`--no-consolidate` to opt out) and lists groups in the prompt ("group of 3, 4 m wide"), at most 6 objects per class | `test_command_parser.py` |
| 32 | A group is approached at its edge (stand-off + its radius), not its centre | `test_a_group_is_approached_...` |
| 33 | *For future recordings* (needs GroundingDINO + a bag; not run here): per-frame NMS within and across classes, image-filling / sliver boxes dropped (`detector.filter_detections`); evidence counted in distinct camera **views**; far observations (stereo error grows with range squared) weigh less and are ignored beyond `--max-object-range 8`; raw observations saved (`*.observations.json`) so `--from-observations` re-clusters without the bag; the raw graph is kept next to the consolidated one | `test_detector_filter.py`, `test_scene_graph.py`, `test_build_scene_graph.py` |

### Saved results in the repository

`saved_state/hospital/` now holds the hospital scene's saved map (`my_map.yaml`, `my_map.pgm`) and the detected objects with their
locations (`scene_graph.json`, raw, plus `scene_graph.consolidated.json`), with a README; `.gitignore` lets exactly these
files through. `docs/media/11_saved_map_hospital.png` shows the map. The warehouse pictures (`02_saved_map.png`,
`03_scene_graph_on_map.png`) stay with the warehouse report, `docs/RESULTS.md`.

### Evidence stored in the repository

`evaluation/` now holds the scene dump (`hospital_prims.json.gz`), the ground truth built from it, the raw benchmark results
(final run and the first partial attempt) and the live-test outputs, with a README. `scripts/dump_stage_prims.py` (runs inside
Isaac Sim) and `scripts/build_ground_truth.py` make the ground truth; `scripts/analyze_scene_graph.py` prints the measurements
behind the perception findings; `scripts/plan_clearance.py` is the probe behind the inflation change. While writing the
analysis script two claims were corrected in these notes: 54 of the 125 non-wall entries (not 67 of 149) share a spot with
another class, and unexplored space was not by itself a sign of a false entry.

### Tooling

`scripts/launch_isaac.sh`, `scripts/isaac_open_scene.py`, `scripts/check_map_alignment.py`, `scripts/eval_scene_graph.py`,
`scripts/plot_scene_graph.py` (the two new figures in `docs/media/`, 09 and 10; nothing existing was removed),
`scripts/live_command_test.py` (5 typed commands through Qwen + Nav2 + the simulator),
`scripts/destination_benchmark.py` (easy / medium / hard destinations), `pipeline.sh start` picks `static` when only the
saved map exists. Tests: 139 -> 238; flake8 clean. Live results: `docs/EVALUATION.md` (5 typed commands 5/5; 5 easy + 5 medium + 5 hard destinations 15/15).

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
