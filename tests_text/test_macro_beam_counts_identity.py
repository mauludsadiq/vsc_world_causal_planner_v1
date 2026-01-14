import json
import subprocess
from pathlib import Path


def test_macro_beam_counts_identity(tmp_path: Path):
    out = tmp_path / "macro_counts.json"
    cmd = [
        "python",
        "-m",
        "experiments.text_block_macro_beam_bench",
        str(out),
        "0",      # seed
        "0.15",   # epsilon
        "40",     # depth
        "32",     # beam
        "4",      # macro_len
        "6",      # topM
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    d = json.loads(out.read_text(encoding="utf-8"))
    assert "BLOCK_MACRO_BEAM_BENCH" in d
    assert "BLOCK_MACRO_BEAM_SEARCH" in d

    s = d["BLOCK_MACRO_BEAM_SEARCH"]
    c = int(s["n_candidates_total"])
    k = int(s["n_kept_total"])
    r = int(s["n_rejected_total"])
    assert c == k + r
