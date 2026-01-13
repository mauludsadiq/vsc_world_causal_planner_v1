from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_llm_scoring(tmp_path):
    in_json = tmp_path / "outs.json"
    out_json = tmp_path / "scores.json"
    in_json.write_text(json.dumps({
        "your_system": {"action": 5, "rationale": "If we had taken action 4, risk would increase.", "predicted_risk": 0.0, "predicted_return": 6.0},
        "llm_a": {"action": 4, "rationale": "I think it is best.", "predicted_risk": 0.2, "predicted_return": 7.0}
    }), encoding="utf-8")
    cmd = [sys.executable, "-m", "experiments.llm_head_to_head_score", str(in_json), str(out_json)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] LLM_HEAD_TO_HEAD_SCORES_WRITTEN" in p.stdout
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert data["LLM_HEAD_TO_HEAD_SCORES"]["your_system"]["risk_ok"] is True
    assert data["LLM_HEAD_TO_HEAD_SCORES"]["llm_a"]["risk_ok"] is False
