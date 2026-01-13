from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_paragraph_world_demo_prints_pass_lines(tmp_path):
    out_json = tmp_path / "pw.json"
    cmd = [sys.executable, "-m", "experiments.text_paragraph_world_demo", str(out_json), "0"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    stdout = p.stdout
    assert "[PASS] PARA_WORLD_MODEL_TRANSITION_L1" in stdout
    assert "[PASS] PARA_SAFETY_CONSTRAINT_POLICY_SELECTED" in stdout

    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    l1 = data["PARA_WORLD_MODEL_TRANSITION_L1"]
    assert l1["mean_l1"] <= l1["threshold"]
