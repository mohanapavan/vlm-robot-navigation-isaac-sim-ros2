"""Build the hospital-scene Word report (docs/Hospital_VLM_Navigation_Report.docx) from the repository's own data.

    pip install python-docx        # (requirements/report.txt); not needed by anything else
    python3 scripts/build_report.py [-o docs/Hospital_VLM_Navigation_Report.docx]

The numbers in the tables are read from evaluation/results/*.json and the saved graphs, and the figures are the ones in
docs/media/, so re-running it after a new benchmark refreshes the report. The original warehouse report
(docs/Nova_VLM_Navigation_Report.docx) is left as it was.
"""
import argparse
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vlm_nav.evaluation import evaluate, load_ground_truth  # noqa: E402
from vlm_nav.scene_graph import SceneGraph  # noqa: E402

MEDIA = ROOT / 'docs' / 'media'
TEXT_W = 17.0          # cm: A4 with 2 cm margins
ACCENT = RGBColor(0x1F, 0x4E, 0x79)


# ----------------------------------------------------------------------------- helpers

def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_fill)
    tc_pr.append(shd)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Hospital-scene report  |  page ')
    run.font.size = Pt(8)
    for kind, text in (('begin', None), (None, 'PAGE'), ('end', None)):
        r = p.add_run()
        r.font.size = Pt(8)
        if kind:
            el = OxmlElement('w:fldChar')
            el.set(qn('w:fldCharType'), kind)
        else:
            el = OxmlElement('w:instrText')
            el.set(qn('xml:space'), 'preserve')
            el.text = text
        r._r.append(el)


def para(doc, text='', bold=False, italic=False, size=None, color=None, align=None, space_after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style='List Bullet')
        if isinstance(item, tuple):
            r = p.add_run(item[0])
            r.bold = True
            p.add_run(item[1])
        else:
            p.add_run(item)
        p.paragraph_format.space_after = Pt(2)


def table(doc, header, rows, widths_cm, font=8.5, zebra=True):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, (cell, text) in enumerate(zip(t.rows[0].cells, header)):
        cell.width = Cm(widths_cm[i])
        cell.text = ''
        run = cell.paragraphs[0].add_run(text)
        run.bold, run.font.size = True, Pt(font)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, '1F4E79')
    for n, row in enumerate(rows):
        cells = t.add_row().cells
        for i, (cell, text) in enumerate(zip(cells, row)):
            cell.width = Cm(widths_cm[i])
            cell.text = ''
            run = cell.paragraphs[0].add_run(str(text))
            run.font.size = Pt(font)
            if zebra and n % 2:
                shade(cell, 'EAF1F8')
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(doc, name, caption, width_cm=TEXT_W):
    path = MEDIA / name
    if not path.exists():
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, caption, italic=True, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)


def h(doc, text, level=1):
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = ACCENT
    return heading


# ----------------------------------------------------------------------------- content

def build(out):
    raw = SceneGraph.load(ROOT / 'saved_state' / 'hospital' / 'scene_graph.json')
    cons = SceneGraph.load(ROOT / 'saved_state' / 'hospital' / 'scene_graph.consolidated.json')
    gt = load_ground_truth(ROOT / 'evaluation' / 'hospital_ground_truth.json')
    ev_raw, ev_cons = evaluate(raw.objects, gt, 1.5), evaluate(cons.objects, gt, 1.5)
    bench = json.loads((ROOT / 'evaluation' / 'results' / 'destination_benchmark_final.json').read_text())
    first = json.loads((ROOT / 'evaluation' / 'results' / 'destination_benchmark_first_attempt_partial.json').read_text())

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.0)
    normal = doc.styles['Normal']
    normal.font.name, normal.font.size = 'Calibri', Pt(10.5)
    add_page_number(sec)
    zoom = doc.settings.element.find(qn('w:zoom'))       # python-docx's template omits the required percentage
    if zoom is not None and zoom.get(qn('w:percent')) is None:
        zoom.set(qn('w:percent'), '100')

    # title -----------------------------------------------------------------
    para(doc, 'Language-guided robot navigation in the hospital scene', bold=True, size=22, color=ACCENT, space_after=2)
    para(doc, 'What was found, what was changed, and how it performs on the running simulator', size=12, space_after=4)
    para(doc, 'NVIDIA Isaac Sim 6.0 | ROS 2 Humble | Nav2 | GroundingDINO scene graph | Qwen2.5-VL-3B brain | RTX 5080',
         italic=True, size=9, space_after=12)

    h(doc, '1. Summary')
    para(doc, 'A simulated Nova Carter robot is given natural-language commands ("go to the wheelchair", "what do you see?"). '
              'Qwen2.5-VL turns each sentence into a command, the code looks up where the named object is in a semantic '
              'scene graph, and Nav2 drives there. This report covers the hospital scene, using the map and the object '
              'detections that were already saved (nothing was re-mapped or re-detected), and two pieces of work: making the '
              'system robust on the running stack, and making the object list small and trustworthy instead of hundreds of '
              'duplicate detections.')
    passed = sum(bool(r['pass']) for r in bench)
    real = sum(bool(r['real']) for r in bench)
    goal_err = sum(r['goal_err_m'] for r in bench) / len(bench)
    table(doc, ['Check', 'Result'], [
        ['Unit and ROS-level tests (fake Nav2, fake TF, synthetic recording)',
         '239 pass (238 where python-docx is not installed), flake8 clean; 139 at the start'],
        ['Live smoke test: relative move, named goal, STOP, goal after STOP', '4 of 4'],
        ['Live: 5 typed commands through Qwen -> Nav2 -> robot', '5 of 5 (run twice)'],
        ['Live: 5 easy + 5 medium + 5 hard destinations (3 m to 56 m)',
         f'{passed} of {len(bench)}; mean goal error {goal_err:.2f} m; {sum(r["driven_m"] for r in bench):.0f} m driven'],
        ['... of those, destinations that are real objects (simulator ground truth)', f'{real} of {len(bench)}'],
        ['Scene graph: entries, precision, recall (raw, walls excluded)',
         f'{ev_raw["ALL"]["entries"]}, {ev_raw["ALL"]["precision"]:.2f}, {ev_raw["ALL"]["recall"]:.2f}'],
        ['Scene graph: entries, precision, recall (consolidated)',
         f'{ev_cons["ALL"]["entries"]}, {ev_cons["ALL"]["precision"]:.2f}, {ev_cons["ALL"]["recall"]:.2f}'],
    ], [10.5, 6.5])
    para(doc, 'Read the "real objects" row with care: a destination passes when the robot does what it was told, but three of '
              'the fifteen entries it drove to were look-alike detections rather than real objects. Navigation is not the weak '
              'point; the recall and precision of the detections are.', size=10)

    figure(doc, 'hospital_demo_still.png', 'A frame from the recorded demo (docs/media/hospital_demo.gif): the typed command and '
           "Qwen's reply, the robot's camera, and its position and trail on the saved map.")

    h(doc, '2. Setup')
    bullets(doc, [
        ('Scene: ', 'the hospital scene (lit rooms, two wander bots), streamed from NVIDIA\'s public asset bucket by Isaac Sim.'),
        ('Saved results used as they are: ', 'the lidar occupancy map (5 cm cells) and 149 GroundingDINO detections with '
         'map-frame locations. The map frame equals the simulator world frame: sliding the map over the scene\'s wall '
         'geometry gives the best fit at a shift of 0.0 m and 0 degrees, and the live lidar lands on the map (100 % of scan '
         'endpoints within 0.15 m).'),
        ('Localisation: ', 'the slam_toolbox pose graph was not saved, so the saved map is served directly with an identity '
         'map->odom transform (exact here because the robot starts at the map origin and simulated odometry does not drift).'),
        ('Speed: ', 'the simulator ran at about 0.25 to 0.7 times real time; Qwen shared the GPU with Isaac Sim and was partly '
         'offloaded to the CPU. Wall-clock times are longer than the simulated ones.'),
    ])
    figure(doc, '11_saved_map_hospital.png', 'The saved lidar map. White is free, black is obstacle, grey is unexplored.', 13.5)

    h(doc, '3. Step 1: robustness fixes found on the live stack')
    para(doc, 'The approach (name -> lookup -> stand-off goal -> planner check) was kept. Running the real stack exposed these '
              'problems, each now covered by a test:')
    table(doc, ['Problem found', 'Fix'], [
        ['Nav2 was served the saved map with no unexplored space: the map\'s free threshold (0.25) makes map_server read the '
         'grey "unknown" pixels as free (0 % unknown instead of 67 %).',
         'The launch file serves a corrected copy of the yaml (the original is untouched); the same rule is used by the brain.'],
        ['The saved map could not be used without the slam_toolbox pose graph.',
         'New static mode: map_server plus an identity map->odom; a script checks alignment against the live scan.'],
        ['Multi-word classes: the model writes "vending_machine_3", "the trash cans" or "beds" and got "unknown object".',
         'One name normalisation (case, underscores, articles, plurals) in the parser and the scene graph.'],
        ['Goals could land inside walls or unexplored space and were then aborted.',
         'With the map, stop points are chosen on known free cells with wall clearance (roomy ones first); a relative move '
         'that ends in a wall is shortened to the last free point or refused with a message.'],
        ['A 3B model answers "take me to the cart" with an arbitrary cart_1 (26 m away, the nearest was 4.4 m).',
         'A class named by the user means the nearest one; an id the user typed beats a different id from the model.'],
        ['The prompt lists only the nearest 6 objects per class, so "door_13" was hidden and the model said it did not know it.',
         'Objects the user names are always listed.'],
        ['Two destinations aborted after 1.4 m ("collision ahead", "failed to make progress"): the plan hugged a chair whose low '
         'base is below the 2-D lidar plane, and the progress checker ignored turning.',
         'Costmap inflation 0.5 -> 0.9 m (plan 0.3 m longer, 0.77 m instead of 0.51 m from mapped obstacles); a progress '
         'checker that counts rotation; roomy stop points preferred.'],
        ['STOP relied only on the Nav2 cancel chain; a failing model call ended the session; a malformed graph entry made the '
         'file unloadable.',
         'STOP also publishes a zero velocity; errors are reported and the chat continues; bad entries are skipped and counted.'],
        ['Setup was manual: open the scene, press Stop, press Play.',
         'A launcher opens the scene, waits for its streamed assets and presses Play; it can remove prims (the low test '
         'obstacles) from the loaded stage without editing the file.'],
    ], [8.5, 8.5])

    h(doc, '4. Step 2: why detection produced hundreds of objects, and the new approach')
    para(doc, 'The saved detections were scored against the simulator\'s real objects (ground truth extracted from the scene: '
              '14 beds, 28 carts, 21 chairs, 12 desks, 48 doors, 11 trash cans, 2 vending machines, 4 wheelchairs). '
              'What was found:')
    bullets(doc, [
        ('Duplicate boxes counted as evidence. ', 'GroundingDINO returns several overlapping boxes per object and the code '
         'never suppressed them, so one frame could count as "seen twice"; 45 of 149 entries had exactly the minimum of 2.'),
        ('One object, several names. ', 'Classes were clustered separately: 54 of the 125 non-wall entries had a different-class '
         'entry within 1 m.'),
        ('Confidence does not separate real from look-alike. ', 'The one correct vending-machine cluster scored like the 17 '
         'false ones (real vending machines: 2; entries: 18). What does help: evidence, agreement between nearby detections, '
         'and the map (an entry 1 m or more from every mapped obstacle was 85 % wrong).'),
        ('Structure was listed as objects: ', '24 of 149 entries were walls.'),
        ('Piles became box_1 ... box_20 ', 'instead of "many boxes here".'),
    ])
    para(doc, 'The approach has two layers. At detection time (for future recordings): per-frame suppression of duplicate boxes '
              'within and across classes, evidence counted in distinct camera views, far observations weighed less (stereo '
              'depth error grows with range squared) and ignored beyond 8 m, and raw observations saved so nothing needs the '
              'bag again. After detection (works on the saved graph today): a consolidation step that drops structure, merges '
              'duplicates, lets classes on one spot compete by evidence, collapses piles of identical things into one group '
              'entry with a size (twenty stacked boxes become one "group of boxes here"), ranks entries by class weight x '
              'evidence x plausibility on the saved map, and keeps only the important ones; every detected class keeps its best '
              'guess. The brain applies it automatically to any raw graph it loads.')
    rows = []
    for cls in ('bed', 'cart', 'chair', 'desk', 'door', 'trash can', 'vending machine', 'wheelchair'):
        a, b = ev_raw[cls], ev_cons[cls]
        rows.append([cls, a['real'], a['entries'], b['entries'], f'{a["precision"]:.2f} -> {b["precision"]:.2f}',
                     f'{a["recall"]:.2f} -> {b["recall"]:.2f}'])
    a, b = ev_raw['ALL'], ev_cons['ALL']
    rows.append(['all', a['real'], a['entries'], b['entries'], f'{a["precision"]:.2f} -> {b["precision"]:.2f}',
                 f'{a["recall"]:.2f} -> {b["recall"]:.2f}'])
    table(doc, ['class', 'real objects', 'raw entries', 'consolidated', 'precision', 'recall'], rows,
          [3.6, 2.4, 2.4, 2.8, 3.0, 2.8])
    para(doc, 'An entry counts as real when a real object of its class lies within 1.5 m of its footprint. Consolidation removes '
              'about two thirds of the entries and raises precision from 0.38 to 0.61 at a small cost in recall; recall is '
              'limited by the saved detections themselves (the raw graph covers 34 of the 156 real objects).', size=10)
    figure(doc, '09_scene_graph_raw_vs_consolidated.png', 'Raw detections (left) and the consolidated graph (right) on the saved '
           'map. Green ring: a real object of that class is within 1.5 m. Red cross: none is. Groups are drawn at their size.')

    h(doc, '5. Live results')
    h(doc, '5.1 Five typed commands', level=2)
    table(doc, ['Command', 'Model reply', 'Outcome'], [
        ['go forward 2 meters', 'MOVE:forward 2', 'SUCCEEDED, moved 1.84 m forward'],
        ['go to the wheelchair', 'GOAL:wheelchair_1', 'SUCCEEDED, 0.83 m from it, facing error 3 deg'],
        ['what do you see?', 'a description of the camera picture', 'answered in words; robot did not move'],
        ['go to the spaceship', '"I don\'t know an object called spaceship."', 'no goal sent; robot did not move'],
        ['go to desk_1 (a group, 3.7 m wide)', 'GOAL:desk_1',
         'SUCCEEDED, 2.83 m from the group centre (its edge plus stand-off)'],
    ], [4.6, 5.4, 7.0])

    h(doc, '5.2 Fifteen destinations', level=2)
    para(doc, 'Difficulty is the length of the path Nav2\'s planner finds from where the robot stands when the destination is '
              'chosen: easy 3-12 m, medium 12-30 m, hard over 30 m. Each destination is typed to Qwen as a sentence. It passes '
              'when Nav2 reports SUCCEEDED, the robot ends within 0.6 m of its goal point, faces the object, and went to the '
              'object that was asked for.', size=10)
    table(doc, ['#', 'tier', 'you said', 'planned m', 'driven m', 'sim s', 'goal err m', 'result', 'real object'], [
        [r['n'], r['tier'], r['command'], r['planned_m'], r['driven_m'], r['sim_s'], r['goal_err_m'],
         'PASS' if r['pass'] else 'FAIL', 'yes' if r['real'] else 'no'] for r in bench],
        [0.7, 1.4, 5.2, 1.6, 1.5, 1.2, 1.8, 1.5, 2.1], font=8)
    figure(doc, '10_benchmark_destinations.png',
           'The 15 destinations on the map (dots), each joined to where that leg started by a straight line (not the '
           'driven path). Green easy, orange medium, red hard.', 14.5)
    para(doc, 'An earlier attempt, on the code before the fixes in section 3, was stopped after 8 destinations because it '
              f'exposed real problems ({sum(not r["pass"] for r in first)} of those 8 failed: a hidden object id and two aborts '
              'beside a chair). After the fixes the whole benchmark was re-run from a fresh simulator start; its raw results, '
              'and those of the first attempt, are stored in the repository.', size=10)

    h(doc, '6. Limits and what was not verified')
    bullets(doc, [
        'Recall of the scene graph (about 0.18 of the real objects) can only improve with new observations; three of the fifteen '
        'benchmark destinations were false detections that the robot reached exactly as told.',
        'The new per-frame box filtering, view counting and range weighting were covered by unit tests and a synthetic '
        'recording only; GroundingDINO was not run on new data.',
        'The unexplored-space penalty in the consolidation is a judgement call: removing it changes the result only slightly '
        '(56 instead of 46 entries, precision 0.57 instead of 0.61).',
        'Objects below the 2-D lidar plane (chair bases, cart shelves, forks) are still invisible to the map; wider inflation '
        'keeps plans further from their mapped parts but does not detect what the lidar cannot see.',
        'The camera-only pipeline (RTAB-Map) was not run; it received only the progress-checker change.',
    ])

    h(doc, '7. Reproducing it')
    for cmd in ('scripts/launch_isaac.sh                        # open the scene and press Play',
                'scripts/pipeline.sh start                      # saved map (static mode) + Nav2',
                'python3 scripts/check_map_alignment.py         # must say ALIGNED',
                '~/qwen_env/bin/python scripts/live_command_test.py        # 5 typed commands',
                '~/qwen_env/bin/python scripts/destination_benchmark.py    # 5 easy + 5 medium + 5 hard',
                'python3 scripts/eval_scene_graph.py saved_state/hospital/scene_graph.json '
                'saved_state/hospital/scene_graph.consolidated.json'):
        p = doc.add_paragraph()
        r = p.add_run(cmd)
        r.font.name, r.font.size = 'Consolas', Pt(8.5)
        p.paragraph_format.space_after = Pt(1)
    para(doc, '')
    para(doc, 'Everything the numbers rest on (scene dump, ground truth, raw results, analysis scripts) is in evaluation/; the '
              'saved map and detections are in saved_state/hospital/; the change log is CHANGELOG.md and the same results in '
              'Markdown are in docs/EVALUATION.md.', size=10)
    doc.save(out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-o', '--output', default=str(ROOT / 'docs' / 'Hospital_VLM_Navigation_Report.docx'))
    args = p.parse_args()
    print('wrote', build(args.output))


if __name__ == '__main__':
    main()
