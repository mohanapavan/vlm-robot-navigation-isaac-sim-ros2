"""Score a scene graph against ground truth (pure python): are the entries real, and is every real thing covered?

Ground truth is a list of `{class, name, min: [x, y, z], max: [x, y, z]}` boxes in the map frame
(`evaluation/hospital_ground_truth.json` was extracted from the simulator scene).

An entry counts as a true positive when its label's class has a real object within `tol` metres of it. Detections are
points on the surface facing the camera, so the tolerance is measured to the object's footprint, not its centre. A
group covers every real object of its class within its own radius + `tol`.
"""
import json
import math
from collections import Counter, defaultdict


def load_ground_truth(path):
    with open(path) as f:
        return json.load(f)['objects']


def distance_to_footprint(x, y, box):
    """Distance from a point to the (x, y) rectangle of a ground-truth box; 0 when inside."""
    dx = max(box['min'][0] - x, 0.0, x - box['max'][0])
    dy = max(box['min'][1] - y, 0.0, y - box['max'][1])
    return math.hypot(dx, dy)


def evaluate(objects, ground_truth, tol=1.5, classes=None):
    """Per-class and overall precision / recall / compactness of `objects` (SceneObjects) against `ground_truth`.

    precision      entries that match a real object of their class / entries
    recall         real objects covered by at least one entry / real objects
    entries        how many things the robot is told about (fewer is easier for the language model)
    per_real       entries per real object covered (1.0 = one entry per thing; above 1 = the same thing is listed twice)
    """
    gt_by_class = defaultdict(list)
    for g in ground_truth:
        gt_by_class[g['class']].append(g)
    classes = classes or sorted(gt_by_class)
    report, confusion = {}, Counter()
    tot_entries = tot_tp = tot_gt = tot_cov = 0
    for cls in classes:
        gts = gt_by_class[cls]
        preds = [o for o in objects if o.label == cls]
        covered, tp, spurious = set(), 0, []
        for o in preds:
            reach = tol + (o.radius if getattr(o, 'kind', 'instance') == 'group' else 0.0)
            hit = [i for i, g in enumerate(gts) if distance_to_footprint(o.x, o.y, g) <= reach]
            if hit:
                tp += 1
                covered.update(hit)
            else:
                spurious.append(o.id)
                near = [(distance_to_footprint(o.x, o.y, g), g['class']) for c, lst in gt_by_class.items()
                        if c != cls for g in lst]
                near = [c for d, c in near if d <= tol]
                confusion[(cls, near and Counter(near).most_common(1)[0][0] or 'nothing real nearby')] += 1
        report[cls] = {
            'entries': len(preds), 'true_positive': tp, 'real': len(gts), 'covered': len(covered),
            'precision': tp / len(preds) if preds else float('nan'),
            'recall': len(covered) / len(gts) if gts else float('nan'),
            'per_real': tp / len(covered) if covered else float('nan'),
            'spurious': spurious,
        }
        tot_entries += len(preds)
        tot_tp += tp
        tot_gt += len(gts)
        tot_cov += len(covered)
    report['ALL'] = {'entries': tot_entries, 'true_positive': tot_tp, 'real': tot_gt, 'covered': tot_cov,
                     'precision': tot_tp / tot_entries if tot_entries else float('nan'),
                     'recall': tot_cov / tot_gt if tot_gt else float('nan'),
                     'per_real': tot_tp / tot_cov if tot_cov else float('nan'), 'spurious': []}
    report['_confusion'] = {f'{a} -> {b}': n for (a, b), n in confusion.most_common()}
    return report


def format_report(report, title=''):
    lines = [title] if title else []
    lines.append(f"{'class':16s} {'entries':>7s} {'real':>5s} {'precision':>9s} {'recall':>7s} {'entries/real':>12s}")
    for cls, r in report.items():
        if cls.startswith('_'):
            continue
        lines.append(f"{cls:16s} {r['entries']:7d} {r['real']:5d} {r['precision']:9.2f} {r['recall']:7.2f} {r['per_real']:12.2f}")
    return '\n'.join(lines)
