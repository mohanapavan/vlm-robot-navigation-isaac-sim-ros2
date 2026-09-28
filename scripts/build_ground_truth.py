"""Turn a stage dump (scripts/dump_stage_prims.py) into the ground-truth object boxes used by the evaluation.

    python3 scripts/build_ground_truth.py [evaluation/hospital_prims.json.gz] [-o evaluation/hospital_ground_truth.json]

Each top-level prop of the scene is assigned to a detection class by its prim name (RULES below); walls, floors, lights and
trim are not objects. World frame == map frame for this scene (the robot spawns at the origin with yaw 0).
Run without arguments it rebuilds the committed file, and `tests/test_ground_truth.py` checks that it matches.
"""
import argparse
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = [   # (class, regex on the prim name); the first match wins
    ('bed', r'^(SM_HospitalBed|Geo_M_Bed|SM_Gurney)'),
    ('cart', r'^(SM_SupplyCart|SM_Cart|SM_GasCart)'),
    ('chair', r'^SM_Chair'),
    ('desk', r'^(SM_Desk|SM_ReceptionDesk)'),
    ('door', r'^(SM_Door_|SM_Door\d|SM_Door$|Geo_M_DobleDoor|GlassDoor)'),
    ('trash can', r'^(trashcan|SM_TrashCan|SM_DisposalStand)'),
    ('vending machine', r'^BP_DrinksMachine'),
    ('wheelchair', r'^SM_WheelChair'),
    ('computer', r'^SM_Computer'),
]
NOTE = ('Hospital scene (scene/slam.usd) ground truth: world = map frame (robot spawns at the origin, yaw 0). '
        'World-space bounding boxes of the top-level SM_* prims of /World/hospital, extracted with Isaac Sim.')


def classify(name):
    """Detection class of a prim name, or None."""
    for cls, pattern in RULES:
        if re.match(pattern, name):
            return cls
    return None


def build(rows, top_depth=3):
    objects = []
    for r in rows:
        if r['depth'] != top_depth:
            continue
        name = r['path'].split('/')[-1]
        cls = classify(name)
        if cls:
            objects.append({'class': cls, 'name': name, 'min': [round(v, 2) for v in r['min']],
                            'max': [round(v, 2) for v in r['max']]})
    return {'note': NOTE, 'objects': objects}


def load_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dump', nargs='?', default=str(ROOT / 'evaluation' / 'hospital_prims.json.gz'))
    p.add_argument('-o', '--output', default=str(ROOT / 'evaluation' / 'hospital_ground_truth.json'))
    args = p.parse_args()
    result = build(load_rows(args.dump))
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=0)
    counts = {}
    for o in result['objects']:
        counts[o['class']] = counts.get(o['class'], 0) + 1
    print(f'wrote {args.output}: {len(result["objects"])} objects {counts}')


if __name__ == '__main__':
    main()
