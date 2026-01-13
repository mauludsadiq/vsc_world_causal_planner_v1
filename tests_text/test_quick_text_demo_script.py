from __future__ import annotations
import json
import subprocess
from pathlib import Path

def test_quick_text_demo_script_runs(repo_root: Path = Path(__file__).resolve().parents[1]):
    p = subprocess.run(["bash", "tools/quick_text_demo.sh", "0", "8"], cwd=repo_root, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] TEXT_FULL_STACK_SUMMARY_WRITTEN" in p.stdout
    data = json.loads((repo_root / "results/text_full_stack_demo.json").read_text(encoding="utf-8"))
    assert "highlights" in data
