from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_llm_prompt_pack_demo(tmp_path):
    out_json = tmp_path / "pack.json"
    cmd = [sys.executable, "-m", "experiments.llm_head_to_head_prompt_pack", str(out_json), "32"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] LLM_PROMPT_PACK_WRITTEN" in p.stdout
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert "PROMPT_PACK" in data
    assert "prompt" in data["PROMPT_PACK"]
    assert "rubric" in data["PROMPT_PACK"]
