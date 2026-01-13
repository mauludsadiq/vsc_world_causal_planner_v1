from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_document_tradeoff_demo_forces_tradeoff(tmp_path):
    out_json = tmp_path / "tradeoff.json"
    cmd = [sys.executable, "-m", "experiments.text_document_tradeoff_demo", str(out_json), "0"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] DOC_SAFETY_TRADEOFF_FORCED" in p.stdout

    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    d = data["DOC_SAFETY_TRADEOFF_FORCED"]
    assert d["opt_risk"] > d["epsilon"]
    assert d["chosen_risk"] <= d["epsilon"]
