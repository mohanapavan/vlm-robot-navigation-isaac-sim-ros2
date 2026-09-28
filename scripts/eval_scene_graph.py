"""Score a scene_graph.json against the hospital scene's ground truth.

    python3 scripts/eval_scene_graph.py ~/scene_graph/scene_graph.json [--tol 1.5] [--compare other.json]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_nav.evaluation import evaluate, format_report, load_ground_truth  # noqa: E402
from vlm_nav.scene_graph import SceneGraph  # noqa: E402

DEFAULT_GT = Path(__file__).resolve().parents[1] / 'evaluation' / 'hospital_ground_truth.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('graph', nargs='+', help='scene_graph.json file(s); several are printed one after another')
    p.add_argument('--ground-truth', default=str(DEFAULT_GT))
    p.add_argument('--tol', type=float, default=1.5, help='metres from a real object\'s footprint that still counts as a hit')
    args = p.parse_args()
    gt = load_ground_truth(args.ground_truth)
    for path in args.graph:
        graph = SceneGraph.load(path)
        report = evaluate(graph.objects, gt, args.tol)
        print(format_report(report, f'{path}  (tolerance {args.tol} m)'))
        if report['_confusion']:
            print('unmatched entries, by what real object is nearby instead:')
            for k, n in list(report['_confusion'].items())[:8]:
                print(f'   {k}: {n}')
        print()


if __name__ == '__main__':
    main()
