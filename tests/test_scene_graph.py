import json

import pytest

from vlm_nav.scene_graph import (DEFAULT_CLASSES, Observation, SceneGraph, build_caption,
                                 cluster_observations, match_phrase_to_class)


@pytest.mark.parametrize('phrase,expected', [
    ('forklift', 'forklift'),
    ('Forklift', 'forklift'),
    ('boxes', 'box'),
    ('wooden box', 'box'),
    ('', None),
    ('   ', None),
    ('box shelf', None),             # merged phrase: ambiguous, dropped
    ('pallet forklift', None),
    ('table', None),                 # not one of our classes
    ('##lift', None),
    ('bo x', None),
])
def test_phrase_mapping(phrase, expected):
    assert match_phrase_to_class(phrase, DEFAULT_CLASSES) == expected


def test_caption_format():
    assert build_caption(['a', 'b c']) == 'a . b c'


def _obs(label, x, y, score=0.5, n=1):
    return [Observation(label, score, x, y, 0.0) for _ in range(n)]


def test_same_class_objects_are_not_averaged_into_one():
    obs = []
    for x in (0.0, 5.0, 10.0, 15.0, 20.0):
        obs += _obs('box', x, 0.0, n=3)
    objs = cluster_observations(obs, radius=1.0, min_observations=2)
    assert len(objs) == 5
    assert sorted(o.x for o in objs) == [0.0, 5.0, 10.0, 15.0, 20.0]     # real positions, no fake midpoint
    assert {o.id for o in objs} == {f'box_{i}' for i in range(1, 6)}


def test_repeat_sightings_of_one_object_merge_and_average():
    obs = [Observation('forklift', 0.6, 4.0 + d, 2.0, 0.0) for d in (-0.2, 0.0, 0.2, 0.1)]
    (o,) = cluster_observations(obs, radius=1.0, min_observations=2)
    assert o.count == 4 and o.x == pytest.approx(4.02, abs=0.1) and o.y == pytest.approx(2.0)
    assert 0.55 < o.score < 0.65                                       # score is kept


def test_different_labels_at_the_same_place_stay_separate():
    obs = _obs('shelf', 1, 1, n=2) + _obs('box', 1, 1, n=2)
    assert {o.label for o in cluster_observations(obs)} == {'shelf', 'box'}


def test_single_sightings_are_dropped_as_noise():
    obs = _obs('cone', 0, 0, n=1) + _obs('door', 9, 9, n=3)
    assert [o.label for o in cluster_observations(obs, min_observations=2)] == ['door']
    assert len(cluster_observations(obs, min_observations=1)) == 2


def test_clusters_whose_centroids_drift_together_are_merged():
    # Seeds 1.2 m apart start as two clusters; later points pull their centroids to ~0.85 m apart.
    obs = [Observation('box', 0.9, 0.0, 0.0), Observation('box', 0.8, 1.2, 0.0),
           Observation('box', 0.7, 0.5, 0.0), Observation('box', 0.6, 0.9, 0.0)]
    assert len(cluster_observations(obs, radius=1.0, min_observations=2)) == 1
    # ...whereas genuinely separate objects stay separate
    far = [Observation('box', 0.9, 0.0, 0.0), Observation('box', 0.8, 0.1, 0.0),
           Observation('box', 0.7, 3.0, 0.0), Observation('box', 0.6, 3.1, 0.0)]
    assert len(cluster_observations(far, radius=1.0, min_observations=2)) == 2


def _graph():
    return SceneGraph(cluster_observations(
        _obs('box', 0, 0, n=2) + _obs('box', 10, 0, n=2) + _obs('forklift', 5, 5, n=2)))


def test_resolve_by_id_label_and_nearest_instance():
    g = _graph()
    assert g.resolve('forklift').id == 'forklift_1'
    assert g.resolve('FORKLIFT_1').label == 'forklift'
    assert g.resolve('box', robot_xy=(9.0, 0.0)).x == 10.0
    assert g.resolve('box', robot_xy=(-1.0, 0.0)).x == 0.0
    assert g.resolve('spaceship') is None
    assert 'box_1' in g.names() and 'box' in g.names()


def test_save_load_round_trip(tmp_path):
    p = tmp_path / 'sg.json'
    _graph().save(p)
    g = SceneGraph.load(p)
    assert g.frame_id == 'map' and [o.id for o in g.objects] == ['box_1', 'box_2', 'forklift_1']


def test_old_format_is_rejected_with_a_clear_message(tmp_path):
    p = tmp_path / 'old.json'
    p.write_text(json.dumps({'forklift': {'x': 1.0, 'y': 2.0, 'z': 0.0, 'count': 3}}))
    with pytest.raises(ValueError, match='Rebuild'):
        SceneGraph.load(p)


def test_class_radius_keeps_one_big_object_as_one_instance():
    # a forklift seen from several distances: surface points 1.8 m apart
    obs = [Observation('forklift', 0.6, x, 0.0, 0.0) for x in (0.0, 0.0, 0.9, 1.8, 1.8)]
    assert len(cluster_observations(obs, radius=1.0, min_observations=2)) == 2
    (o,) = cluster_observations(obs, radius=1.0, min_observations=2, class_radius={'forklift': 2.0})
    assert o.count == 5
    # other labels keep the default radius
    boxes = [Observation('box', 0.6, x, 0.0, 0.0) for x in (0.0, 0.0, 1.8, 1.8)]
    assert len(cluster_observations(boxes, radius=1.0, min_observations=2, class_radius={'forklift': 2.0})) == 2


def test_resolve_all_orders_instances_by_distance():
    g = _graph()                                     # box_1 at (0,0), box_2 at (10,0), forklift_1 at (5,5)
    assert [o.id for o in g.resolve_all('box', (9.0, 0.0))] == ['box_2', 'box_1']
    assert [o.id for o in g.resolve_all('Box_1')] == ['box_1']
    assert g.resolve_all('spaceship') == []
