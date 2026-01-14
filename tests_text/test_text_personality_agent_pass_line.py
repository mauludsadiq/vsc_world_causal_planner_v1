from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_personality_agent_schema(tmp_path):
    out_json = tmp_path / "pa.json"
    cmd = [sys.executable, "-m", "experiments.text_personality_agent_demo", str(out_json), "0", "0.15"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] PERSONALITY_CONTRACT_WRITTEN" in p.stdout
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    d = data["PERSONALITY_CONTRACT"]
    assert "main_reply" in d
    assert "safety_verdict" in d
    assert "mandatory_counterfactual" in d
