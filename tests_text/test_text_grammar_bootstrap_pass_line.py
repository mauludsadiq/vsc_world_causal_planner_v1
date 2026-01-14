from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_grammar_bootstrap_accepts_roundtrip(tmp_path):
    out_json = tmp_path / "gb.json"
    cmd = [sys.executable, "-m", "experiments.text_grammar_bootstrap_demo", str(out_json), "compose_v0", "40"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] GRAMMAR_BOOTSTRAP_RULE_ACCEPTED" in p.stdout
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert data["GRAMMAR_BOOTSTRAP"]["roundtrip"]["ok"] is True
