from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_block_beam_search(tmp_path):
    out_json = tmp_path / "beam.json"
    cmd = [sys.executable, "-m", "experiments.text_block_beam_search_demo", str(out_json), "0", "0.15", "6", "6"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] BLOCK_BEAM_SEARCH_WRITTEN" in p.stdout
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert "BLOCK_BEAM_SEARCH" in data
    assert len(data["BLOCK_BEAM_SEARCH"]["best_path"]) == 6
