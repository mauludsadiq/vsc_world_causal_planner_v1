from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_block_complex_sweep_demo(tmp_path):
    out_json = tmp_path / "sweep.json"
    cmd = [sys.executable, "-m", "experiments.text_block_complex_sweep_demo", str(out_json), "0"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] BLOCK_COMPLEX_SWEEP_WRITTEN" in p.stdout
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert len(data["BLOCK_COMPLEX_SWEEP"]["rows"]) >= 2
