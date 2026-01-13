from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_block_world_demo_prints_pass_line(tmp_path):
    out_json = tmp_path / "bw.json"
    cmd = [sys.executable, "-m", "experiments.text_block_world_demo", str(out_json), "0", "8"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] BLOCK_WORLD_MODEL_TRANSITION_L1" in p.stdout

    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    d = data["BLOCK_WORLD_MODEL_TRANSITION_L1"]
    assert d["mean_l1"] <= d["threshold"]
