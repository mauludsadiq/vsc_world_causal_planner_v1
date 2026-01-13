from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_text_sentence_demo_prints_pass_lines(tmp_path):
    out_json = tmp_path / "text_sentence_demo.json"
    cmd = [sys.executable, "-m", "experiments.text_sentence_demo", str(out_json), "0"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    stdout = p.stdout
    assert "[PASS] TEXT_SCM_BACKDOOR" in stdout
    assert "[PASS] TEXT_WORLD_MODEL_TRANSITION_L1" in stdout
    assert "[PASS] TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE" in stdout
    assert "[PASS] TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED" in stdout

    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    scm = data["TEXT_SCM_BACKDOOR"]
    assert scm["max_abs_err_backdoor"] <= scm["tol"]
    assert scm["min_gap_naive"] >= 0.07

    l1 = data["TEXT_WORLD_MODEL_TRANSITION_L1"]
    assert l1["mean_l1"] <= l1["threshold"]

    pl = data["TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE"]
    assert pl["abs_return_diff"] <= 1e-6
