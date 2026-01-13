from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_real_text_probe(tmp_path):
    out_json = tmp_path / "rt.json"
    cmd = [sys.executable, "-m", "experiments.text_real_input_full_stack_probe", str(out_json), "corpora/real_text_samples.txt", "0"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] REAL_TEXT_PROBE_WRITTEN" in p.stdout
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert data["REAL_TEXT_PROBE"]["n_lines"] >= 2
