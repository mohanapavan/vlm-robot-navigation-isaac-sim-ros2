import json

import pytest

from vlm_nav.scene_graph import (DEFAULT_CLASSES, Observation, SceneGraph, SceneObject, build_caption,
                                 cluster_observations, dump_observations, load_observations, match_phrase_to_class,
                                 normalize_name)


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


# ---------------------------------------------------------------- names: '_' / ' ' / case / articles / plurals

def _hospital_graph():
    return SceneGraph([SceneObject('vending machine_1', 'vending machine', 1.0, 0.0, 1.0, 5, 0.5),
                       SceneObject('trash can_1', 'trash can', 5.0, 0.0, 0.5, 4, 0.5),
                       SceneObject('trash can_2', 'trash can', 9.0, 0.0, 0.5, 4, 0.5),
                       SceneObject('box_1', 'box', 2.0, 2.0, 0.2, 4, 0.5),
                       SceneObject('bed_1', 'bed', -3.0, 0.0, 0.5, 4, 0.5)])


@pytest.mark.parametrize('spoken,expected_id', [
    ('vending machine_1', 'vending machine_1'),
    ('vending_machine_1', 'vending machine_1'),
    ('Vending-Machine 1', 'vending machine_1'),
    ('the vending machine', 'vending machine_1'),
    ('trash_can_2', 'trash can_2'),
    ('the closest trash cans', 'trash can_1'),
    ('boxes', 'box_1'),
    ('beds.', 'bed_1'),
    ('  BED  ', 'bed_1'),
])
def test_names_are_matched_however_the_model_spells_them(spoken, expected_id):
    assert _hospital_graph().resolve(spoken, robot_xy=(0.0, 0.0)).id == expected_id


def test_unknown_names_still_resolve_to_nothing():
    g = _hospital_graph()
    assert g.resolve('spaceship') is None and g.resolve('') is None and g.resolve_all('the') == []
    assert normalize_name(' Trash_Can-2. ') == 'trash can 2'


# ---------------------------------------------------------------- loading

def test_version_2_graphs_still_load_and_get_default_fields(tmp_path):
    path = tmp_path / 'g.json'
    path.write_text(json.dumps({'version': 2, 'frame_id': 'map', 'objects': [
        {'id': 'bed_1', 'label': 'bed', 'x': 1.0, 'y': 2.0, 'z': 0.5, 'count': 3, 'score': 0.4}]}))
    g = SceneGraph.load(path)
    o = g.objects[0]
    assert (o.kind, o.members, o.size, o.z_min, o.z_max) == ('instance', 1, 0.0, 0.5, 0.5) and g.consolidated is False


def test_malformed_entries_are_skipped_not_fatal(tmp_path):
    path = tmp_path / 'g.json'
    good = {'id': 'bed_1', 'label': 'bed', 'x': 1.0, 'y': 2.0, 'z': 0.5, 'count': 3, 'score': 0.4}
    path.write_text(json.dumps({'version': 3, 'objects': [
        good, {'id': 'bed_2', 'label': 'bed', 'x': 'left', 'y': 0, 'z': 0, 'count': 1, 'score': 1},
        {'id': 'bed_3', 'label': 'bed'}, dict(good, id='bed_4', x=float('nan')), dict(good, id='bed_5', extra_key=1)]}))
    g = SceneGraph.load(path)
    assert [o.id for o in g.objects] == ['bed_1', 'bed_5'] and g.dropped_on_load == 3


@pytest.mark.parametrize('content', ['[]', '{"version": 1, "objects": []}', '{"objects": []}'])
def test_unusable_files_are_rejected_with_a_clear_error(tmp_path, content):
    path = tmp_path / 'g.json'
    path.write_text(content)
    with pytest.raises(ValueError, match='scene graph'):
        SceneGraph.load(path)


# ---------------------------------------------------------------- evidence = distinct views, far depth counts less

def test_several_boxes_in_one_frame_are_one_sighting():
    same_frame = [Observation('box', 0.6, 1.0 + 0.05 * i, 1.0, 0.3, frame=7, range=3.0) for i in range(6)]
    assert cluster_observations(same_frame, min_observations=2) == []
    two_frames = same_frame[:3] + [Observation('box', 0.6, 1.0, 1.0, 0.3, frame=8, range=3.0)]
    (o,) = cluster_observations(two_frames, min_observations=2)
    assert o.count == 2


def test_observations_without_frame_ids_count_one_each_as_before():
    assert cluster_observations(_obs('box', 1.0, 1.0, n=2), min_observations=2)[0].count == 2


def test_far_observations_weigh_less_in_the_position():
    near = Observation('box', 0.5, 0.0, 0.0, 0.3, frame=1, range=2.0)
    far = Observation('box', 0.5, 0.9, 0.0, 0.3, frame=2, range=14.0)       # stereo depth this far is unreliable
    (o,) = cluster_observations([near, far], radius=1.5, min_observations=2)
    assert o.x < 0.3                                                       # pulled toward the close-up sighting
    (o,) = cluster_observations([Observation('box', 0.5, 0.0, 0.0, 0.3), Observation('box', 0.5, 0.9, 0.0, 0.3)],
                                radius=1.5, min_observations=2)
    assert o.x == pytest.approx(0.45, abs=0.01)                            # no ranges: plain average as before


def test_observations_round_trip_through_json(tmp_path):
    obs = [Observation('bed', 0.5, 1.0, 2.0, 0.6, frame=3, range=4.5), Observation('door', 0.7, -1.0, 0.0)]
    dump_observations(obs, tmp_path / 'o.json')
    assert load_observations(tmp_path / 'o.json') == obs
