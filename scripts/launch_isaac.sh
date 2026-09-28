#!/bin/bash
# Start Isaac Sim, open a scene and press Play (so /clock, /scan, the cameras and /tf are live).
#
#   scripts/launch_isaac.sh [scene.usd]      # default: scene/slam.usd in this repository
#
# The scene's low test obstacles (/World/TestObstacles) are removed from the loaded stage; the .usd file itself is not
# edited. Set REMOVE_PRIMS="" to keep them, or REMOVE_PRIMS="/World/A,/World/B" for others.
# Extra arguments after the scene go to Isaac Sim, e.g.  --no-window  for a headless run.
# Log: ~/pipeline_logs/isaac.log
ISAAC_DIR="${ISAAC_DIR:-$HOME/isaacsim_standalone}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$(realpath "${1:-$REPO/scene/slam.usd}")"
[ $# -gt 0 ] && shift
LOGS="$HOME/pipeline_logs"
REMOVE_PRIMS="${REMOVE_PRIMS-/World/TestObstacles}"

[ -f "$SCENE" ] || { echo "scene not found: $SCENE"; exit 1; }
[ -x "$ISAAC_DIR/isaac-sim.sh" ] || { echo "Isaac Sim not found in $ISAAC_DIR (set ISAAC_DIR)"; exit 1; }
if pgrep -f "kit/kit .*isaacsim.exp.full.kit" >/dev/null; then
    echo "Isaac Sim is already running; close it first (its stage would not be reloaded)"; exit 1
fi
mkdir -p "$LOGS"
cd "$ISAAC_DIR" || exit 1
nohup ./isaac-sim.sh --/renderer/activeGpu=0 --/vlm_nav/scene="$SCENE" --/vlm_nav/remove="$REMOVE_PRIMS" \
    --exec "$REPO/scripts/isaac_open_scene.py" "$@" > "$LOGS/isaac.log" 2>&1 &
echo "Isaac Sim starting with $SCENE (log: $LOGS/isaac.log)"
