from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_full_stack_demo_runs_and_writes_summary(tmp_path):
    out_json = tmp_path / "full.json"
    cmd = [sys.executable, "-m", "experiments.text_full_stack_demo", str(out_json), "0", "8"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    stdout = p.stdout
    assert "[PASS] TEXT_SCM_BACKDOOR" in stdout
    assert "[PASS] PARA_WORLD_MODEL_TRANSITION_L1" in stdout
    assert "[PASS] DOC_SAFETY_TRADEOFF_FORCED" in stdout
    assert "[PASS] BLOCK_WORLD_MODEL_TRANSITION_L1" in stdout
    assert "[PASS] BLOCK_SAFETY_TRADEOFF_FORCED" in stdout
    assert "[PASS] TEXT_FULL_STACK_SUMMARY_WRITTEN" in stdout

    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    bt = data["highlights"]["BLOCK_SAFETY_TRADEOFF_FORCED"]
    assert bt["forced"] is True
    assert bt["opt_risk"] > bt["epsilon"]
    assert bt["chosen_risk"] <= bt["epsilon"]
