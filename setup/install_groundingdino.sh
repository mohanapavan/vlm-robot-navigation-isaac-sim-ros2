#!/bin/bash
# Install GroundingDINO (open-vocabulary object detector) + weights into a venv that can also see ROS 2.
# Usage:  source /opt/ros/humble/setup.bash && bash setup/install_groundingdino.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${PERCEPTION_ENV:-$HOME/perception_env}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$HOME/weights}"

if ! python3 -c "import rclpy" 2>/dev/null; then
    echo "rclpy not importable. Run:  source /opt/ros/humble/setup.bash" >&2
    exit 1
fi

# --system-site-packages: rclpy, tf2_ros, rosbag2_py and cv2 come from the ROS install (they are not on PyPI).
python3 -m venv --system-site-packages "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
# The apt setuptools that --system-site-packages exposes is too old for editable installs (needs >= 64).
pip install "setuptools>=64,<80" wheel "packaging>=23"
pip install -r "$REPO_DIR/requirements/perception.txt"

cd "$HOME"
[ -d GroundingDINO ] || git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
# --no-build-isolation: build against the torch installed above. The CUDA op (_C) is only compiled when
# nvcc / CUDA_HOME is available; without it the detector falls back to CPU.
pip install --no-build-isolation -c "$REPO_DIR/requirements/constraints.txt" -e .

mkdir -p "$WEIGHTS_DIR" "$HOME/scene_graph"
cd "$WEIGHTS_DIR"
[ -f groundingdino_swint_ogc.pth ] || wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
[ -f GroundingDINO_SwinT_OGC.py ] || wget -q https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

# sanity check
python3 -c "
from groundingdino.util.inference import load_model
load_model('$WEIGHTS_DIR/GroundingDINO_SwinT_OGC.py', '$WEIGHTS_DIR/groundingdino_swint_ogc.pth', device='cpu')
print('GroundingDINO model loaded successfully')
"
echo "Done. Activate with:  source $ENV_DIR/bin/activate"
