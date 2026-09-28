"""The ground truth and the analysis can be rebuilt from what is committed."""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import analyze_scene_graph  # noqa: E402
import build_ground_truth  # noqa: E402

from vlm_nav.map_grid import MapGrid  # noqa: E402
from vlm_nav.scene_graph import SceneGraph, SceneObject  # noqa: E402


def test_prim_names_map_to_detection_classes():
    c = build_ground_truth.classify
    assert c('SM_HospitalBed_02d3') == 'bed' and c('SM_Gurney_01a') == 'bed'
    assert c('SM_SupplyCart_01e5') == 'cart' and c('SM_Chair_04a2') == 'chair'
    assert c('SM_ReceptionDesk_01a') == 'desk' and c('BP_DrinksMachine_2') == 'vending machine'
    assert c('SM_Door_01c') == 'door' and c('GlassDoor2') == 'door' and c('trashcan3') == 'trash can'
    assert c('SM_WheelChair_01a3') == 'wheelchair'
    for not_an_object in ('Geo_M1_SideWall_5', 'Geo_M1_Floor', 'Light_test15', 'SM_PillBottleSet_01a', 'Geo_M_DoorFrame3'):
        assert c(not_an_object) is None


def test_only_top_level_props_become_ground_truth():
    rows = [{'path': '/World/hospital/SM_Chair_01a', 'depth': 3, 'min': [0, 0, 0], 'max': [1, 1, 1.234]},
            {'path': '/World/hospital/SM_Chair_01a/Mesh', 'depth': 4, 'min': [0, 0, 0], 'max': [1, 1, 1]},
            {'path': '/World/hospital/Geo_M1_Floor', 'depth': 3, 'min': [0, 0, 0], 'max': [9, 9, 0]}]
    (obj,) = build_ground_truth.build(rows)['objects']
    assert obj == {'class': 'chair', 'name': 'SM_Chair_01a', 'min': [0, 0, 0], 'max': [1, 1, 1.23]}


def test_committed_ground_truth_is_exactly_what_the_committed_dump_gives():
    rebuilt = build_ground_truth.build(build_ground_truth.load_rows(ROOT / 'evaluation' / 'hospital_prims.json.gz'))
    assert rebuilt == json.loads((ROOT / 'evaluation' / 'hospital_ground_truth.json').read_text())
    assert len(rebuilt['objects']) == 156


def test_the_saved_hospital_results_load_and_match_the_documented_numbers():
    graph = SceneGraph.load(ROOT / 'saved_state' / 'hospital' / 'scene_graph.json')
    assert len(graph) == 149 and not graph.consolidated
    grid = MapGrid.from_yaml(ROOT / 'saved_state' / 'hospital' / 'my_map.yaml')
    gt = json.loads((ROOT / 'evaluation' / 'hospital_ground_truth.json').read_text())['objects']
    s = analyze_scene_graph.summarize(graph.objects, grid, gt)
    assert (s['entries'], s['structure'], s['minimum_evidence']) == (149, 24, 45)
    assert s['other_class_within_1m'] == 54 and s['hit_rate'] == 0.38
    assert s['hit_rate_by']['clearance >= 1.0 m']['hit_rate'] == 0.15
    assert (grid.cells == -1).mean() > 0.65                                  # 67 % of the map is unexplored
    done = SceneGraph.load(ROOT / 'saved_state' / 'hospital' / 'scene_graph.consolidated.json')
    assert len(done) == 46 and done.consolidated and any(o.kind == 'group' for o in done.objects)


def test_result_files_are_complete():
    rows = json.loads((ROOT / 'evaluation' / 'results' / 'destination_benchmark_final.json').read_text())
    assert [r['tier'] for r in rows] == ['easy'] * 5 + ['medium'] * 5 + ['hard'] * 5 and all(r['pass'] for r in rows)
    first = json.loads((ROOT / 'evaluation' / 'results' / 'destination_benchmark_first_attempt_partial.json').read_text())
    assert len(first) == 8 and sum(not r['pass'] for r in first) == 3
    with gzip.open(ROOT / 'evaluation' / 'hospital_prims.json.gz', 'rt') as f:
        assert len(json.load(f)) == 5031


def test_analysis_handles_an_empty_graph():
    s = analyze_scene_graph.summarize([SceneObject('bed_1', 'bed', 0, 0, 0.5, 2, 0.4)])
    assert s['entries'] == 1 and s['minimum_evidence'] == 1 and s['same_class_duplicates'] == {}
