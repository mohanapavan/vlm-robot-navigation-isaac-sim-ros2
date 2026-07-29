#!/bin/bash
# Install GroundingDINO (open-vocabulary object detector) + weights.
set -e

mkdir -p ~/scene_graph
pip install groundingdino-py 2>/dev/null || true

cd ~
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

mkdir -p ~/weights
cd ~/weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

# sanity check
cd ~
python3 -c "
from groundingdino.util.inference import load_model
model = load_model('weights/GroundingDINO_SwinT_OGC.py', 'weights/groundingdino_swint_ogc.pth')
print('Model loaded successfully')
"
