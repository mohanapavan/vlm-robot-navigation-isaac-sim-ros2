#!/bin/bash
# Start / stop the lidar SLAM + Nav2 stack in the background (logs in ~/pipeline_logs).
#
#   scripts/pipeline.sh start [static|localization|mapping]
#       static        serve the saved map (~/my_map.yaml) with an identity map->odom; no pose graph needed  (default
#                     when ~/my_map.yaml exists and there is no ~/my_map_posegraph.posegraph)
#       localization  resume slam_toolbox from the saved pose graph (~/my_map_posegraph.posegraph)
#       mapping       build a new map
#   scripts/pipeline.sh stop
#   scripts/pipeline.sh status
#
# Isaac Sim must already be playing scene/slam.usd (scripts/launch_isaac.sh; after Stop -> Play the robot is at the
# map origin). Restart this after every Stop/Play: sim time jumps back to 0. RVIZ=false skips RViz.
# Order matters: SLAM first, then Nav2 (see docs/commands.md, troubleshooting).
LOGS="$HOME/pipeline_logs"
if [ -n "${2:-}" ]; then
    MODE="$2"
elif [ -f "$HOME/my_map_posegraph.posegraph" ]; then
    MODE=localization
elif [ -f "$HOME/my_map.yaml" ]; then
    MODE=static
else
    MODE=mapping
fi

source_ros() {
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    [ -f "$HOME/ws/install/setup.bash" ] && source "$HOME/ws/install/setup.bash"
}

launch_pids() { ps -eo pid,args | grep -E "^ *[0-9]+ /usr/bin/python3 /opt/ros/humble/bin/ros2 launch vlm_nav (slam|navigation)" | awk '{print $1}'; }
node_pids() { ps -eo pid,args | grep -E "^ *[0-9]+ /opt/ros/humble/lib/(nav2_|slam_toolbox|rviz2|tf2_ros)" | awk '{print $1}'; }

stop_all() {
    local p
    p=$(launch_pids); [ -n "$p" ] && kill -INT $p 2>/dev/null
    for _ in $(seq 1 12); do [ -z "$(node_pids)$(launch_pids)" ] && break; sleep 1; done
    p="$(launch_pids) $(node_pids)"; [ -n "${p// /}" ] && { kill -TERM $p 2>/dev/null; sleep 4; }
    p="$(launch_pids) $(node_pids)"; [ -n "${p// /}" ] && kill -KILL $p 2>/dev/null
    sleep 1
    echo "stopped (launch processes left: $(launch_pids | wc -l), ROS nodes left: $(node_pids | wc -l))"
}

case "${1:-}" in
    start)
        [ -n "$(launch_pids)$(node_pids)" ] && { echo "already running; use 'stop' first"; exit 1; }
        source_ros
        mkdir -p "$LOGS"
        if [ "$MODE" = "localization" ] && [ ! -f "$HOME/my_map_posegraph.posegraph" ]; then
            echo "no ~/my_map_posegraph.posegraph: use 'start static' (saved map only) or 'start mapping'"; exit 1
        fi
        if [ "$MODE" = "static" ] && [ ! -f "$HOME/my_map.yaml" ]; then
            echo "no ~/my_map.yaml: copy saved_state/hospital/my_map.* to ~ (see saved_state/hospital/README.md) or run in mapping mode first"; exit 1
        fi
        nohup ros2 launch vlm_nav slam_lidar.launch.py mode:="$MODE" rviz:="${RVIZ:-true}" > "$LOGS/slam.log" 2>&1 &
        echo "SLAM ($MODE) starting..."; sleep 25
        nohup ros2 launch vlm_nav navigation.launch.py sensor:=lidar > "$LOGS/nav2.log" 2>&1 &
        for _ in $(seq 1 60); do grep -q "Managed nodes are active" "$LOGS/nav2.log" 2>/dev/null && break; sleep 3; done
        if grep -q "Managed nodes are active" "$LOGS/nav2.log" 2>/dev/null; then
            echo "READY: SLAM ($MODE) + Nav2 are up. Logs: $LOGS"
        else
            echo "Nav2 did not become active; see $LOGS/nav2.log"; exit 1
        fi ;;
    stop) stop_all ;;
    status)
        echo "launch processes: $(launch_pids | wc -l) | ROS nodes: $(node_pids | wc -l)"
        [ -f "$LOGS/nav2.log" ] && echo "Nav2 active: $(grep -c 'Managed nodes are active' "$LOGS/nav2.log") | TF errors: $(grep -c tf_help "$LOGS/nav2.log")" ;;
    *) sed -n '2,10p' "$0"; exit 1 ;;
esac
