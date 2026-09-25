#!/bin/bash
# Set up an isolated env for the Qwen2.5-VL robot brain (kept separate from GroundingDINO, which needs an
# older transformers). The venv can see the ROS 2 install so rclpy works; rclpy is NOT installable from PyPI.
# Usage:  source /opt/ros/humble/setup.bash && bash setup/install_qwen.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${QWEN_ENV:-$HOME/qwen_env}"

if ! python3 -c "import rclpy" 2>/dev/null; then
    echo "rclpy not importable. Run:  source /opt/ros/humble/setup.bash" >&2
    exit 1
fi

python3 -m venv --system-site-packages "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
# The apt setuptools/packaging that --system-site-packages exposes are too old for current wheels.
pip install "setuptools>=64,<80" wheel "packaging>=23"
pip install -r "$REPO_DIR/requirements/brain.txt"

# Download the model now (about 7 GB) instead of on first use.
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct')
print('Qwen2.5-VL-3B-Instruct downloaded')
PY

echo "Done. Activate with:  source $ENV_DIR/bin/activate  (after sourcing /opt/ros/humble/setup.bash)"
