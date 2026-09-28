import json
import math

import numpy as np
import pytest

from vlm_nav import consolidate as C
from vlm_nav.map_grid import OCCUPIED, UNKNOWN, MapGrid
from vlm_nav.scene_graph import SceneGraph, SceneObject


def obj(i, label, x, y, z=0.6, count=5, score=0.5):
    return SceneObject(f'{label}_{i}', label, x, y, z, count, score)


def kinds(objs):
    return sorted((o.label, o.kind) for o in objs)


def test_walls_are_structure_not_destinations():
    out, rep = C.consolidate([obj(1, 'wall', 0, 0, count=30), obj(1, 'door', 5, 5)])
    assert [o.label for o in out] == ['door'] and rep.structure == ['wall_1']


def test_fragments_of_one_object_are_merged_and_evidence_adds_up():
    out, rep = C.consolidate([obj(1, 'bed', 0.0, 0.0, count=4), obj(2, 'bed', 0.9, 0.3, count=3),
                              obj(3, 'bed', 20.0, 0.0, count=6)])
    beds = sorted(out, key=lambda o: o.x)
    assert len(beds) == 2 and rep.merged == 1
    assert beds[0].count == 7 and beds[0].x == pytest.approx(0.4, abs=0.3)      # evidence-weighted centre
    assert beds[0].kind == 'instance' and beds[0].size > 0                      # merged: it is now known to be wider


def test_twenty_stacked_boxes_are_one_entry_not_box_1_to_box_20():
    """The scenario from the brief: many boxes piled up in one place."""
    rng = np.random.default_rng(1)
    pile = [obj(i, 'box', 5 + rng.normal(0, 0.4), 2 + rng.normal(0, 0.4), z=0.2 + 0.25 * (i % 5), count=3)
            for i in range(20)]
    out, rep = C.consolidate(pile)
    assert len(out) == 1
    (g,) = out
    assert g.label == 'box' and g.id == 'box_1'
    assert g.count == 60 and g.members >= 3            # all the evidence, however it was merged
    assert g.z_min < 0.3 and g.z_max > 0.9              # the pile is tall
    assert (g.x, g.y) == pytest.approx((5, 2), abs=0.6)


def test_a_pile_becomes_a_group_and_far_apart_boxes_stay_separate():
    pile = [obj(i, 'box', 5 + 0.7 * i, 2, count=3) for i in range(5)]           # 2.8 m long row
    lone = [obj(9, 'box', 40, 40, count=6), obj(10, 'box', -30, 10, count=6)]
    out, _ = C.consolidate(pile + lone)
    assert kinds(out) == [('box', 'group'), ('box', 'instance'), ('box', 'instance')]
    g = next(o for o in out if o.kind == 'group')
    assert g.size > 2.8 and g.members == 5 and g.radius == pytest.approx(g.size / 2)


def test_two_neighbours_are_not_a_group():
    out, _ = C.consolidate([obj(1, 'chair', 0, 0), obj(2, 'chair', 2.0, 0)])
    assert kinds(out) == [('chair', 'instance'), ('chair', 'instance')]


def test_groups_never_stretch_beyond_the_maximum_size():
    row = [obj(i, 'chair', 1.6 * i, 0, count=4) for i in range(12)]            # 17.6 m of chairs, each within the gap
    out, _ = C.consolidate(row)
    assert all(o.size <= C.Settings().max_group_size + C.CLASS_PROFILES['chair'].footprint for o in out)
    assert len(out) >= 3


def test_doors_are_never_grouped_even_when_dense():
    doors = [obj(i, 'door', 3.0 * i, 0, count=5) for i in range(6)]
    out, _ = C.consolidate(doors)
    assert len(out) == 6 and all(o.kind == 'instance' for o in out)


def test_one_spot_two_classes_the_better_supported_wins():
    out, rep = C.consolidate([obj(1, 'vending machine', 3.0, 3.0, count=8, score=0.6),
                              obj(1, 'desk', 3.2, 3.1, count=3, score=0.4)])
    assert [o.label for o in out] == ['vending machine']
    assert rep.arbitrated[0][1] == 'desk' and rep.arbitrated[0][3] == 'vending machine'


def test_neighbouring_classes_farther_apart_both_survive():
    out, _ = C.consolidate([obj(1, 'bed', 0, 0, count=8), obj(1, 'chair', 1.5, 0, count=8)])
    assert sorted(o.label for o in out) == ['bed', 'chair']


def test_weak_evidence_is_dropped_but_a_class_keeps_its_best_guess():
    weak = [obj(1, 'bed', 0, 0, count=2, score=0.35), obj(2, 'bed', 30, 0, count=2, score=0.32)]
    strong = [obj(1, 'door', 5, 5, count=9, score=0.6)]
    out, rep = C.consolidate(weak + strong, settings=C.Settings(min_importance=0.2))
    assert sorted(o.label for o in out) == ['bed', 'door'] and len([o for o in out if o.label == 'bed']) == 1
    assert (min(o.importance for o in out if o.label == 'bed') < 0.2)     # kept as the class's best guess only
    out, _ = C.consolidate(weak + strong, settings=C.Settings(min_importance=0.2, keep_best_per_class=False))
    assert [o.label for o in out] == ['door']


def test_the_size_limit_keeps_every_class_represented():
    many = [obj(i, 'door', 4.0 * i, 0, count=8) for i in range(10)] + [obj(1, 'bed', 200, 200, count=3)]
    out, rep = C.consolidate(many, settings=C.Settings(max_objects=3))
    assert sum(o.label == 'door' for o in out) == 3 and sum(o.label == 'bed' for o in out) == 1
    assert len(rep.over_limit) == 7


def _grid():
    cells = np.zeros((100, 100), dtype=np.int16)
    cells[:, :3] = OCCUPIED                  # wall along x in [0, 0.3)
    cells[:, 60:] = UNKNOWN                  # x >= 6 unexplored
    return MapGrid(cells, 0.1, (0.0, 0.0))


def test_map_plausibility_prefers_confirmed_places():
    near_wall = obj(1, 'cart', 0.5, 5.0, count=6)             # next to the mapped wall
    floating = obj(2, 'cart', 4.0, 5.0, count=6)              # in the open, 3.7 m from anything mapped
    unexplored = obj(3, 'cart', 8.0, 5.0, count=6)            # in unexplored space
    out, _ = C.consolidate([near_wall, floating, unexplored], grid=_grid(), settings=C.Settings(min_importance=0.0))
    imp = {round(o.x): o.importance for o in out}
    assert imp[0] > imp[4] > imp[8]
    out, _ = C.consolidate([near_wall, floating, unexplored], grid=None, settings=C.Settings(min_importance=0.0))
    assert len({o.importance for o in out}) == 1               # without a map they are equally believable


def test_height_outside_the_plausible_range_lowers_importance():
    out, _ = C.consolidate([obj(1, 'desk', 0, 0, z=0.6), obj(2, 'desk', 50, 0, z=0.02)],
                           settings=C.Settings(min_importance=0.0))
    by_x = {round(o.x): o.importance for o in out}
    assert by_x[0] > by_x[50]


def test_unknown_classes_use_default_profile_and_ids_are_snake_case():
    out, _ = C.consolidate([obj(1, 'trash can', 0, 0, count=8), obj(2, 'trash can', 30, 0, count=7),
                            obj(1, 'spaceship', 9, 9, count=8)])
    assert sorted(o.id for o in out) == ['spaceship_1', 'trash_can_1', 'trash_can_2']
    graph = SceneGraph(out)
    assert graph.resolve('the trash cans').label == 'trash can' and graph.resolve('Trash_Can_2') is not None


def test_non_finite_entries_and_empty_input_do_not_crash():
    assert C.consolidate([])[0] == []
    out, _ = C.consolidate([obj(1, 'bed', float('nan'), 0), obj(2, 'bed', 1, 1, count=9)])
    assert len(out) == 1 and math.isfinite(out[0].x)


def test_consolidation_is_deterministic():
    entries = [obj(i, 'chair', (i * 7) % 13, (i * 3) % 11, count=3 + i % 4) for i in range(30)]
    a, _ = C.consolidate(entries)
    b, _ = C.consolidate(list(reversed(entries)))
    assert [(o.id, o.x, o.y) for o in a] == [(o.id, o.x, o.y) for o in b]


def test_saved_graph_round_trips_with_group_fields(tmp_path):
    out, _ = C.consolidate([obj(i, 'box', 0.7 * i, 0, count=3) for i in range(5)])
    path = tmp_path / 'g.json'
    SceneGraph(out).save(path)
    loaded = SceneGraph.load(path)
    assert json.loads(path.read_text())['version'] == 3
    g = loaded.objects[0]
    assert g.kind == 'group' and g.members == 5 and g.size == out[0].size and g.z_max == out[0].z_max


def test_cli_writes_a_consolidated_file(tmp_path, capsys):
    src = tmp_path / 'raw.json'
    SceneGraph([obj(i, 'box', 0.7 * i, 0, count=3) for i in range(5)] + [obj(1, 'wall', 9, 9)]).save(src)
    assert C.main(['--input', str(src), '--map', 'none', '--output', str(tmp_path / 'out.json')]) == 0
    assert 'raw entries -> 1 kept' in capsys.readouterr().out
    assert len(SceneGraph.load(tmp_path / 'out.json')) == 1
