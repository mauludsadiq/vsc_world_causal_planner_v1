from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_visualize_full_stack(tmp_path):
    in_json = Path("results/text_full_stack_demo.json")
    if not in_json.exists():
        return
    out_dir = tmp_path / "viz"
    cmd = [sys.executable, "-m", "experiments.text_visualize_full_stack", str(in_json), str(out_dir)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] TEXT_FULL_STACK_VIZ_WRITTEN" in p.stdout
    viz_json = out_dir / "viz.json"
    data = json.loads(viz_json.read_text(encoding="utf-8"))
    assert (len(data["TEXT_FULL_STACK_VIZ"]["files"]) >= 1) or (data["TEXT_FULL_STACK_VIZ"].get("skipped") is True)
