#!/bin/bash
# Set up an isolated env for the Qwen2.5-VL robot brain.
# (Kept separate from GroundingDINO, which pins an older transformers version.)
set -e

python3 -m venv ~/qwen_env
source ~/qwen_env/bin/activate
pip install torch transformers accelerate qwen-vl-utils rclpy

echo "Done. Activate with:  source ~/qwen_env/bin/activate"
