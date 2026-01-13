from __future__ import annotations
import subprocess
from pathlib import Path

def test_ci_text_stack_smoke_fast_script_runs(repo_root: Path = Path(__file__).resolve().parents[1]):
    p = subprocess.run(["bash", "tools/ci_text_stack_smoke_fast.sh", "0", "8"], cwd=repo_root, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] RUN_ALL_TEXT_DEMOS_DONE" in p.stdout
