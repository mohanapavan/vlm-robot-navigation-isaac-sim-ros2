# Evaluation data

Everything the numbers in `docs/EVALUATION.md` and `CHANGELOG.md` rest on, so each can be checked or redone.

| file | what it is | made by |
|---|---|---|
| `hospital_prims.json.gz` | every prim of the hospital scene with its world-space bounding box (5031 prims) | `scripts/dump_stage_prims.py`, run inside Isaac Sim |
| `hospital_ground_truth.json` | the 156 real objects of the detected classes (beds, carts, chairs, desks, doors, trash cans, vending machines, wheelchairs, computers) | `scripts/build_ground_truth.py` from the dump above; `tests/test_ground_truth.py` checks the file is exactly that output |
| `results/destination_benchmark_final.json` | the 15 destinations (5 easy, 5 medium, 5 hard): sentence, model reply, planned and driven path, time, goal and facing error, pass, real object | `scripts/destination_benchmark.py` |
| `results/destination_benchmark_first_attempt_partial.json` | the first attempt, stopped after 8 destinations: 3 failures (a hidden object id, two aborts beside a chair) that led to fixes 34-36 in the changelog | same script, older code |
| `results/live_command_test_first_run.txt`, `..._final_run.txt` | the 5 typed commands, before and after those fixes | `scripts/live_command_test.py` |

Redo the analysis without the simulator:

```bash
python3 scripts/build_ground_truth.py                                   # rebuild the ground truth from the committed dump
python3 scripts/eval_scene_graph.py saved_state/hospital/scene_graph.json saved_state/hospital/scene_graph.consolidated.json
python3 scripts/analyze_scene_graph.py saved_state/hospital/scene_graph.json      # why the raw graph is so large
python3 scripts/plot_scene_graph.py destinations evaluation/results/destination_benchmark_final.json \
    saved_state/hospital/scene_graph.consolidated.json docs/media/10_benchmark_destinations.png
```

The live results (`results/`) can only be re-measured with the simulator running (`docs/commands.md`, quick start).
World frame = map frame for this scene: the robot spawns at the origin with yaw 0, and the saved map lines up with the scene's
wall geometry at a shift of 0.0 m / 0 degrees.
