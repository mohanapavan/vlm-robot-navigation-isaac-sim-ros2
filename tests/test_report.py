"""The Word report is generated from the repository's own data (needs python-docx; skipped where it is not installed).

The skip is inside the test on purpose: a module-level importorskip made pytest 6 report "1 skipped" for the whole session.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_report_builds_with_the_measured_numbers(tmp_path):
    docx = pytest.importorskip('docx')
    sys.path.insert(0, str(ROOT / 'scripts'))
    import build_report

    doc = docx.Document(build_report.build(str(tmp_path / 'report.docx')))
    cells = [c.text for t in doc.tables for r in t.rows for c in r.cells]
    text = '\n'.join([p.text for p in doc.paragraphs] + cells)
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith('Heading')]
    assert len(headings) >= 9 and headings[0] == '1. Summary'
    for needle in ('15 of 15', '5 of 5', '12 of 15', 'go to door_4', '0.38', '0.61'):
        assert needle in text
    assert len(doc.tables) == 5 and len(doc.inline_shapes) >= 3
    assert cells.count('PASS') == 15 and 'FAIL' not in cells
