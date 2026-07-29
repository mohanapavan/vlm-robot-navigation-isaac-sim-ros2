import json
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import torch
from groundingdino.util.inference import load_model, predict
from torchvision.transforms import functional as TF
from PIL import Image

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

BAG_PATH = "/home/user/warehouse_bag"
WEIGHTS = "/home/user/weights/groundingdino_swint_ogc.pth"
CONFIG = "/home/user/weights/GroundingDINO_SwinT_OGC.py"
OUTPUT = "/home/user/scene_graph/scene_graph.json"
CLASSES = "box . shelf . pallet . forklift . door . wall . crate . container . ladder . cone"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
SAMPLE_EVERY = 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model on {device}...")
model = load_model(CONFIG, WEIGHTS, device=device)
typestore = get_typestore(Stores.ROS2_HUMBLE)

objects = {}
frame_count = 0
processed = 0

print("Reading bag...")
with Reader(BAG_PATH) as reader:
    img_topic = '/front_stereo_camera/left/image_raw'
    odom_topic = '/chassis/odom'
    latest_odom = None

    for connection, timestamp, rawdata in reader.messages():
        topic = connection.topic

        if topic == odom_topic:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            latest_odom = {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            }

        if topic == img_topic:
            frame_count += 1
            if frame_count % SAMPLE_EVERY != 0:
                continue
            if latest_odom is None:
                continue

            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            img_data = np.frombuffer(msg.data, dtype=np.uint8)
            img = img_data.reshape((msg.height, msg.width, 3))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            img_tensor = TF.to_tensor(pil_img)

            boxes, logits, phrases = predict(
                model=model,
                image=img_tensor,
                caption=CLASSES,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=device
            )

            for phrase in phrases:
                if phrase not in objects:
                    objects[phrase] = {'positions': [], 'count': 0}
                objects[phrase]['positions'].append(latest_odom)
                objects[phrase]['count'] += 1

            processed += 1
            print(f"Frame {frame_count} | found: {phrases} | odom: ({latest_odom['x']:.2f}, {latest_odom['y']:.2f})")

scene_graph = {}
for label, data in objects.items():
    positions = data['positions']
    scene_graph[label] = {
        'x': round(float(np.mean([p['x'] for p in positions])), 3),
        'y': round(float(np.mean([p['y'] for p in positions])), 3),
        'z': round(float(np.mean([p['z'] for p in positions])), 3),
        'count': data['count']
    }

with open(OUTPUT, 'w') as f:
    json.dump(scene_graph, f, indent=2)

print(f"\nDone! Processed {processed} frames")
print(json.dumps(scene_graph, indent=2))
