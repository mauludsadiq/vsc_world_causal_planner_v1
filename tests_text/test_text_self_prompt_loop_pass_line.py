from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_self_prompt_loop(tmp_path):
    out_json = tmp_path / "loop.json"
    cmd = [sys.executable, "-m", "experiments.text_self_prompt_loop_demo", str(out_json), "0", "0.15", "6"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] SELF_PROMPT_LOOP_WRITTEN" in (p.stderr + p.stdout)
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert "SELF_PROMPT_LOOP" in data
    assert len(data["SELF_PROMPT_LOOP"]["transcript"]) == 6
