from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

from text_world.state import enumerate_states
from text_world.render_parse_clean import render_sentence_clean, parse_sentence_clean

def test_clean_render_parse_roundtrip_over_all_states():
    for st in enumerate_states():
        txt = render_sentence_clean(st)
        st2 = parse_sentence_clean(txt)
        assert st2 == st

def test_clean_demo_prints_pass_line(tmp_path):
    out_json = tmp_path / "clean_demo.json"
    cmd = [sys.executable, "-m", "experiments.text_sentence_clean_demo", str(out_json)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] TEXT_SENTENCE_CLEAN_RENDER_PARSE_ROUNDTRIP" in p.stdout

    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert data["n_failures"] == 0
    assert data["n_states"] == 96
