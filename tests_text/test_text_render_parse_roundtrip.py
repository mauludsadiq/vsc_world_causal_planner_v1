from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

from text_world.state import enumerate_states
from text_world.render_parse import render_sentence, parse_sentence

def test_render_parse_roundtrip_over_all_states():
    for st in enumerate_states():
        s = render_sentence(st)
        st2 = parse_sentence(s)
        assert st2 == st

def test_render_parse_demo_prints_pass_line(tmp_path):
    out_json = tmp_path / "render_parse.json"
    cmd = [sys.executable, "-m", "experiments.text_sentence_render_parse_demo", str(out_json)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] TEXT_SENTENCE_RENDER_PARSE_ROUNDTRIP" in p.stdout

    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert data["n_failures"] == 0
    assert data["n_states"] == 96
