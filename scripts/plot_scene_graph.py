"""Draw scene graphs (and benchmark destinations) on the saved map, for the docs. Needs only numpy + OpenCV.

    python3 scripts/plot_scene_graph.py compare RAW.json CONSOLIDATED.json OUT.png [--map ~/my_map.yaml]
    python3 scripts/plot_scene_graph.py destinations BENCHMARK.json CONSOLIDATED.json OUT.png [--map ~/my_map.yaml]
    python3 scripts/plot_scene_graph.py map OUT.png [--map ~/my_map.yaml]

`compare`: two panels, the raw detections (walls left out) and the consolidated graph. Groups are drawn as circles of their
size. A green ring means a real object of that class lies within 1.5 m (from evaluation/hospital_ground_truth.json),
a red cross means none does.
`destinations`: each benchmark destination as a numbered dot, joined to where the robot started by a straight line (the
straight line is NOT the driven path); colour = tier.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_nav.evaluation import distance_to_footprint, load_ground_truth  # noqa: E402
from vlm_nav.scene_graph import SceneGraph  # noqa: E402

GT_PATH = Path(__file__).resolve().parents[1] / 'evaluation' / 'hospital_ground_truth.json'
CLASS_BGR = {'bed': (200, 60, 60), 'cart': (0, 140, 255), 'chair': (60, 170, 60), 'desk': (180, 50, 180),
             'door': (40, 40, 220), 'trash can': (170, 170, 0), 'vending machine': (0, 190, 230),
             'wheelchair': (90, 90, 90), 'computer': (120, 80, 30)}
TIER_BGR = {'easy': (60, 170, 60), 'medium': (0, 140, 255), 'hard': (40, 40, 220)}


class Canvas:
    """The saved map cropped to its known area and padded (grey) so every point in `include` fits, with world -> pixel."""

    def __init__(self, yaml_path, scale=1.0, margin=1.5, include=()):
        import yaml
        from PIL import Image
        meta = yaml.safe_load(open(yaml_path))
        image = meta['image'] if os.path.isabs(meta['image']) else os.path.join(os.path.dirname(yaml_path), meta['image'])
        img = np.array(Image.open(image).convert('L'))
        self.res, (self.ox, self.oy) = float(meta['resolution']), meta['origin'][:2]
        self.h, self.scale = img.shape[0], scale
        known = np.argwhere(img != 205)
        pad = int(margin / self.res)
        rows = [known[:, 0].min() - pad, known[:, 0].max() + pad]
        cols = [known[:, 1].min() - pad, known[:, 1].max() + pad]
        for x, y in include:                       # full-image pixel coordinates; may lie outside the map image
            cols += [int((x - self.ox) / self.res) - pad // 2, int((x - self.ox) / self.res) + pad // 2]
            rows += [int(self.h - (y - self.oy) / self.res) - pad // 2, int(self.h - (y - self.oy) / self.res) + pad // 2]
        self.r0, self.r1, self.c0, self.c1 = min(rows), max(rows), min(cols), max(cols)
        canvas = np.full((self.r1 - self.r0, self.c1 - self.c0), 236, np.uint8)
        src_r0, src_r1 = max(self.r0, 0), min(self.r1, img.shape[0])
        src_c0, src_c1 = max(self.c0, 0), min(self.c1, img.shape[1])
        patch = np.where(img[src_r0:src_r1, src_c0:src_c1] == 205, 236, img[src_r0:src_r1, src_c0:src_c1])
        canvas[src_r0 - self.r0:src_r1 - self.r0, src_c0 - self.c0:src_c1 - self.c0] = patch
        self.img = cv2.resize(np.stack([canvas] * 3, axis=-1), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def px(self, x, y):
        col = (x - self.ox) / self.res - self.c0
        row = (self.h - (y - self.oy) / self.res) - self.r0
        return int(col * self.scale), int(row * self.scale)

    def metres(self, m):
        return int(m / self.res * self.scale)


def title(img, text, sub=''):
    bar = np.full((70, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(bar, sub, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def legend(img, labels):
    x = 12
    for label in labels:
        cv2.circle(img, (x + 6, img.shape[0] - 16), 6, CLASS_BGR.get(label, (0, 0, 0)), -1)
        cv2.putText(img, label, (x + 18, img.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        x += 24 + 9 * len(label)
    return img


def draw_graph(canvas, objects, gt, skip=('wall',)):
    img = canvas.img.copy()
    hits = 0
    for o in objects:
        if o.label in skip:
            continue
        reach = 1.5 + (o.radius if o.kind == 'group' else 0.0)
        real = any(distance_to_footprint(o.x, o.y, g) <= reach for g in gt if g['class'] == o.label)
        hits += real
        x, y = canvas.px(o.x, o.y)
        color = CLASS_BGR.get(o.label, (0, 0, 0))
        if o.kind == 'group':
            overlay = img.copy()
            cv2.circle(overlay, (x, y), max(canvas.metres(o.radius), 6), color, -1, cv2.LINE_AA)
            img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
            cv2.circle(img, (x, y), max(canvas.metres(o.radius), 6), color, 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)
        if real:
            cv2.circle(img, (x, y), 8, (0, 170, 0), 2, cv2.LINE_AA)
        else:
            cv2.line(img, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 220), 2, cv2.LINE_AA)
            cv2.line(img, (x - 6, y + 6), (x + 6, y - 6), (0, 0, 220), 2, cv2.LINE_AA)
    return img, hits


def compare(args):
    graphs = [SceneGraph.load(args.first).objects, SceneGraph.load(args.second).objects]
    canvas = Canvas(os.path.expanduser(args.map), scale=args.scale,
                    include=[(o.x, o.y) for objs in graphs for o in objs if o.label != 'wall'])
    gt = load_ground_truth(GT_PATH)
    panels = []
    for objs, name in zip(graphs, ('Raw detections', 'Consolidated')):
        shown = [o for o in objs if o.label != 'wall']
        img, hits = draw_graph(canvas, objs, gt)
        groups = sum(o.kind == 'group' for o in shown)
        sub = f'{len(shown)} entries' + (f' ({groups} groups)' if groups else '') + \
              f', {hits / max(len(shown), 1):.0%} next to a real object of their class (green ring; red cross = none)'
        panels.append(legend(title(img, name, sub), sorted({o.label for o in shown})))
    height = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, height - p.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in panels]
    return np.hstack(panels)


def destinations(args):
    graph = SceneGraph.load(args.second)
    rows = json.load(open(args.first))
    canvas = Canvas(os.path.expanduser(args.map), scale=args.scale,
                    include=[graph.resolve(r['target']).xy for r in rows] + [tuple(r['start']) for r in rows])
    img = canvas.img.copy()
    for r in rows:
        obj = graph.resolve(r['target'])
        color = TIER_BGR[r['tier']]
        sx, sy = canvas.px(*r['start'])
        ex, ey = canvas.px(obj.x, obj.y)
        cv2.line(img, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA)
        cv2.circle(img, (sx, sy), 4, (60, 60, 60), -1, cv2.LINE_AA)
        cv2.circle(img, (ex, ey), 11, color, -1, cv2.LINE_AA)
        cv2.putText(img, str(r['n']), (ex - (5 if r['n'] < 10 else 9), ey + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    passed = sum(bool(r['pass']) for r in rows)
    cv2.putText(img, 'green = easy, orange = medium, red = hard', (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (30, 30, 30), 1, cv2.LINE_AA)
    img = title(img, f'Benchmark destinations: {passed}/{len(rows)} reached',
                'dot = destination (number = order), grey dot = start of that leg, line = straight link, not the '
                'driven path')
    return img


def plain_map(args):
    """The saved map alone, with the origin (where the robot starts) and a scale bar."""
    canvas = Canvas(os.path.expanduser(args.map), scale=args.scale)
    img = canvas.img.copy()
    ox, oy = canvas.px(0.0, 0.0)
    cv2.drawMarker(img, (ox, oy), (0, 160, 0), cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
    cv2.putText(img, 'start / map origin (0, 0)', (ox + 14, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 110, 0), 1, cv2.LINE_AA)
    bar = canvas.metres(10.0)
    y = img.shape[0] - 22
    cv2.line(img, (16, y), (16 + bar, y), (30, 30, 30), 3)
    cv2.putText(img, '10 m', (16, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return title(img, 'Saved lidar map (hospital scene)',
                 'white = free, black = obstacle, grey = unexplored; 5 cm cells; map frame = simulator world frame')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('compare', 'destinations', 'map'):
        s = sub.add_parser(name)
        if name != 'map':
            s.add_argument('first')
            s.add_argument('second')
        s.add_argument('output')
        s.add_argument('--map', default=os.path.join(os.path.expanduser('~'), 'my_map.yaml'))
        s.add_argument('--scale', type=float, default=0.75)
    args = p.parse_args()
    img = {'compare': compare, 'destinations': destinations, 'map': plain_map}[args.cmd](args)
    cv2.imwrite(args.output, img)
    print(f'wrote {args.output} ({img.shape[1]}x{img.shape[0]})')


if __name__ == '__main__':
    main()
