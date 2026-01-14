import json
import subprocess
from pathlib import Path


def test_text_block_hier_demo_runs_and_writes_json(tmp_path):
    out = tmp_path / "text_block_hier_demo.json"
    cmd = ["python", "-m", "experiments.text_block_hier_demo", str(out), "0", "8", "6"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "[PASS] TEXT_BLOCK_HIER_DEMO_WRITTEN" in p.stdout

    data = json.loads(out.read_text(encoding="utf-8"))
    plan = data["TEXT_BLOCK_HIER_DEMO"]["plan"]
    eps = float(plan["epsilon"])
    assert float(plan["risk_max"]) <= eps
