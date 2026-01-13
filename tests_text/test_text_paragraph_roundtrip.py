from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

from text_world.state import enumerate_states
from text_world.paragraph import normalize_paragraph, render_paragraph_clean, parse_paragraph_clean

def test_paragraph_roundtrip_small_deterministic_set():
    states = enumerate_states()
    # deterministic subset: first 10 states -> 1000 combos; sample 30 fixed combos
    picks = states[:10]
    combos = [
        (picks[0], picks[1], picks[2]),
        (picks[3], picks[4], picks[5]),
        (picks[6], picks[7], picks[8]),
        (picks[9], picks[0], picks[1]),
        (picks[2], picks[3], picks[4]),
        (picks[5], picks[6], picks[7]),
        (picks[8], picks[9], picks[0]),
        (picks[1], picks[2], picks[3]),
        (picks[4], picks[5], picks[6]),
        (picks[7], picks[8], picks[9]),
    ]
    for s1, s2, s3 in combos:
        p = normalize_paragraph(s1, s2, s3)
        txt = render_paragraph_clean(p)
        p2 = parse_paragraph_clean(txt)
        assert p2 == p

def test_paragraph_demo_prints_pass_line(tmp_path):
    out_json = tmp_path / "paragraph_demo.json"
    cmd = [sys.executable, "-m", "experiments.text_paragraph_demo", str(out_json), "0"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] TEXT_PARAGRAPH_RENDER_PARSE_ROUNDTRIP" in p.stdout

    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert data["n_failures"] == 0
    assert data["samples"] == 200
