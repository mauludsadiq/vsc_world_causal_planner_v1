import json
import subprocess
from pathlib import Path


def test_block_world_l1_scale_gate(tmp_path):
    out = tmp_path / "l1_n128.json"
    cmd = ["python", "-m", "experiments.text_block_world_demo", str(out), "0", "128", "64", "8", "0.20"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    d = json.loads(out.read_text(encoding="utf-8"))["BLOCK_WORLD_MODEL_TRANSITION_L1"]
    assert float(d["mean_l1"]) <= 0.20
