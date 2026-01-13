from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_block_counterfactual_explain_demo(tmp_path):
    out_json = tmp_path / "cf.json"
    cmd = [sys.executable, "-m", "experiments.text_block_counterfactual_explain_demo", str(out_json), "0", "32", "5", "4"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] BLOCK_COUNTERFACTUAL_EXPLANATION_WRITTEN" in p.stdout
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    ex = data["BLOCK_COUNTERFACTUAL_EXPLANATION"]["explanation"]
    assert "If we had taken action" in ex
    assert "expected risk" in ex
