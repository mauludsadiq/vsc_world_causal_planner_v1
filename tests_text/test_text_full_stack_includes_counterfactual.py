from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_full_stack_includes_counterfactual(tmp_path):
    out_json = tmp_path / "full.json"
    cmd = [sys.executable, "-m", "experiments.text_full_stack_demo", str(out_json), "0", "32"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] BLOCK_COUNTERFACTUAL_EXPLANATION_INCLUDED" in p.stdout
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert "BLOCK_COUNTERFACTUAL_EXPLANATION" in data["highlights"]
    ex = data["highlights"]["BLOCK_COUNTERFACTUAL_EXPLANATION"]["explanation"]
    assert "If we had taken action" in ex
