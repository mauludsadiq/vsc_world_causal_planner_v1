from __future__ import annotations
import subprocess
import sys
from pathlib import Path

def test_run_all_text_demos_script(tmp_path):
    script = Path("tools/run_all_text_demos.sh")
    assert script.exists()

    env = dict(**{k: v for k, v in dict().items()})
    p = subprocess.run(
        ["bash", str(script), "0", "8"],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    assert "[PASS] RUN_ALL_TEXT_DEMOS_DONE" in p.stdout
