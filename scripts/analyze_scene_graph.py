"""Diagnose a raw scene graph: the measurements behind the consolidation design (see CHANGELOG 0.3.0).

    python3 scripts/analyze_scene_graph.py saved_state/hospital/scene_graph.json [--map saved_state/hospital/my_map.yaml]
                                           [--ground-truth evaluation/hospital_ground_truth.json]

Prints: entries per class, how much evidence each entry has, same-class duplicates (nearest-neighbour distances),
different-class entries on one spot, where entries sit on the saved map (free / occupied / unexplored, clearance from mapped
obstacles), and, with a ground truth, how often an entry is a real object depending on each of those (a "hit" is a real
object of the class within --tol metres of the entry's footprint).
"""
import argparse
import collections
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_nav.evaluation import distance_to_footprint, load_ground_truth  # noqa: E402
from vlm_nav.map_grid import FREE, OCCUPIED, MapGrid  # noqa: E402
from vlm_nav.scene_graph import SceneGraph  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STRUCTURE = ('wall',)


def summarize(objects, grid=None, ground_truth=None, tol=1.5):
    """All the diagnostics as a dict (also what the tests look at)."""
    things = [o for o in objects if o.label not in STRUCTURE]
    out = {'entries': len(objects), 'structure': len(objects) - len(things),
           'per_class': dict(collections.Counter(o.label for o in objects)),
           'minimum_evidence': sum(o.count == 2 for o in objects), 'low_evidence': sum(o.count <= 3 for o in objects)}
    same = {}
    for label in sorted({o.label for o in things}):
        pts = np.array([[o.x, o.y] for o in things if o.label == label])
        if len(pts) > 1:
            d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
            np.fill_diagonal(d, 1e9)
            same[label] = {'entries': len(pts), 'nearest_within_1.5m': int((d.min(1) < 1.5).sum()),
                           'median_nearest_m': round(float(np.median(d.min(1))), 2)}
    out['same_class_duplicates'] = same
    out['other_class_within_1m'] = sum(any(q.label != o.label and math.hypot(q.x - o.x, q.y - o.y) < 1.0
                                           for q in things) for o in things)
    if grid is not None:
        state = collections.Counter({FREE: 'free', OCCUPIED: 'occupied'}.get(grid.state(o.x, o.y), 'unexplored') for o in things)
        out['map_state'] = dict(state)
    if ground_truth is not None:
        rows = []
        for o in things:
            gts = [g for g in ground_truth if g['class'] == o.label]
            hit = any(distance_to_footprint(o.x, o.y, g) <= tol for g in gts)
            rows.append((o, hit))
        out['hit_rate'] = round(float(np.mean([h for _, h in rows])), 2) if rows else None

        def rate(pred):
            sel = [h for o, h in rows if pred(o)]
            return {'n': len(sel), 'hit_rate': round(float(np.mean(sel)), 2) if sel else None}
        by = {'count == 2': rate(lambda o: o.count == 2), 'count >= 5': rate(lambda o: o.count >= 5)}
        if grid is not None:
            by['in unexplored space'] = rate(lambda o: grid.state(o.x, o.y) not in (FREE, OCCUPIED))
            by['clearance < 0.3 m'] = rate(lambda o: grid.clearance(o.x, o.y) < 0.3)
            by['clearance >= 1.0 m'] = rate(lambda o: grid.clearance(o.x, o.y) >= 1.0)
        by['another class within 1 m'] = rate(lambda o: any(q.label != o.label and q.label not in STRUCTURE and
                                                            math.hypot(q.x - o.x, q.y - o.y) < 1.0 for q in things))
        out['hit_rate_by'] = by
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('graph')
    p.add_argument('--map', default=str(ROOT / 'saved_state' / 'hospital' / 'my_map.yaml'))
    p.add_argument('--ground-truth', default=str(ROOT / 'evaluation' / 'hospital_ground_truth.json'))
    p.add_argument('--tol', type=float, default=1.5)
    args = p.parse_args()
    grid = MapGrid.from_yaml(args.map) if Path(args.map).is_file() else None
    gt = load_ground_truth(args.ground_truth) if Path(args.ground_truth).is_file() else None
    s = summarize(SceneGraph.load(args.graph).objects, grid, gt, args.tol)
    print(f"{s['entries']} entries, {s['structure']} of them structure (walls); per class: {s['per_class']}")
    print(f"evidence: {s['minimum_evidence']} entries have exactly 2 observations, {s['low_evidence']} have 3 or fewer")
    print('same-class duplicates (nearest same-class entry closer than 1.5 m):')
    for label, d in s['same_class_duplicates'].items():
        print(f"   {label:16s} {d['nearest_within_1.5m']:3d} of {d['entries']:3d}   median nearest {d['median_nearest_m']} m")
    print(f"entries with a different-class entry within 1 m: {s['other_class_within_1m']}")
    if 'map_state' in s:
        print(f"position on the saved map: {s['map_state']}")
    if 'hit_rate' in s:
        print(f"overall: {s['hit_rate']:.0%} of entries are next to a real object of their class; by property:")
        for name, r in s['hit_rate_by'].items():
            print(f"   {name:26s} n={r['n']:3d}   real: {r['hit_rate']}")


if __name__ == '__main__':
    main()
