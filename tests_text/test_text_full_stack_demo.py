from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_full_stack_demo_runs_and_writes_summary(tmp_path):
    out_json = tmp_path / "full.json"
    cmd = [sys.executable, "-m", "experiments.text_full_stack_demo", str(out_json), "0"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    stdout = p.stdout
    assert "[PASS] TEXT_SCM_BACKDOOR" in stdout
    assert "[PASS] PARA_WORLD_MODEL_TRANSITION_L1" in stdout
    assert "[PASS] DOC_SAFETY_TRADEOFF_FORCED" in stdout
    assert "[PASS] TEXT_FULL_STACK_SUMMARY_WRITTEN" in stdout

    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert "artifacts" in data and "highlights" in data
    assert data["highlights"]["DOC_SAFETY_TRADEOFF_FORCED"]["opt_risk"] > data["highlights"]["DOC_SAFETY_TRADEOFF_FORCED"]["epsilon"]
    assert data["highlights"]["DOC_SAFETY_TRADEOFF_FORCED"]["chosen_risk"] <= data["highlights"]["DOC_SAFETY_TRADEOFF_FORCED"]["epsilon"]
