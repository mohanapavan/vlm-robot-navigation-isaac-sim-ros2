import math

from vlm_nav.evaluation import distance_to_footprint, evaluate
from vlm_nav.scene_graph import SceneObject

BOX = {'class': 'bed', 'name': 'b', 'min': [0, 0, 0], 'max': [2, 1, 1]}


def test_distance_to_footprint():
    assert distance_to_footprint(1, 0.5, BOX) == 0.0
    assert distance_to_footprint(5, 0.5, BOX) == 3.0
    assert math.isclose(distance_to_footprint(5, 5, BOX), math.hypot(3, 4))


def test_precision_recall_and_duplicates():
    gt = [BOX, {'class': 'bed', 'name': 'c', 'min': [10, 0, 0], 'max': [12, 1, 1]}]
    objs = [SceneObject('bed_1', 'bed', 1.0, 0.5, 0.5, 3, 0.5),         # on the first bed
            SceneObject('bed_2', 'bed', 2.5, 0.5, 0.5, 3, 0.5),         # duplicate of the first bed (within 1.5 m)
            SceneObject('bed_3', 'bed', 50, 50, 0.5, 3, 0.5)]           # nowhere near anything
    r = evaluate(objs, gt, tol=1.5)['bed']
    assert (r['entries'], r['true_positive'], r['covered'], r['real']) == (3, 2, 1, 2)
    assert r['precision'] == 2 / 3 and r['recall'] == 0.5 and r['per_real'] == 2.0
    assert r['spurious'] == ['bed_3']


def test_a_group_covers_everything_inside_its_radius():
    gt = [{'class': 'bed', 'name': str(i), 'min': [i * 3.0, 0, 0], 'max': [i * 3.0 + 1, 1, 1]} for i in range(3)]
    group = SceneObject('bed_1', 'bed', 3.5, 0.5, 0.5, 9, 0.5, kind='group', members=3, size=8.0)
    r = evaluate([group], gt, tol=0.5)['bed']
    assert r['covered'] == 3 and r['recall'] == 1.0 and r['per_real'] == 1 / 3


def test_wrong_class_is_reported_as_confusion():
    gt = [BOX, {'class': 'door', 'name': 'd', 'min': [20, 0, 0], 'max': [21, 1, 1]}]
    rep = evaluate([SceneObject('door_1', 'door', 1.0, 0.5, 0.5, 3, 0.5)], gt, tol=1.5)
    assert rep['_confusion'] == {'door -> bed': 1}
