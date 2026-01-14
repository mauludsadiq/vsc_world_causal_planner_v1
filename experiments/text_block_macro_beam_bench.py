from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from text_world.agent.block_macro_search import run_block_macro_beam_search


def main():
    out_json = "results/sweeps/beam/macro_beam.json"
    seed = 0
    epsilon = 0.15
    depth = 40
    beam = 16
    macro_len = 2
    topM = 16

    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    if len(sys.argv) >= 4:
        epsilon = float(sys.argv[3])
    if len(sys.argv) >= 5:
        depth = int(sys.argv[4])
    if len(sys.argv) >= 6:
        beam = int(sys.argv[5])
    if len(sys.argv) >= 7:
        macro_len = int(sys.argv[6])
    if len(sys.argv) >= 8:
        topM = int(sys.argv[7])

    t0 = time.perf_counter()
    rep = run_block_macro_beam_search(
        out_json,
        seed=seed,
        epsilon=epsilon,
        depth=depth,
        beam=beam,
        macro_len=macro_len,
        topM=topM,
    )
    search = rep.get("BLOCK_MACRO_BEAM_SEARCH", {})
    t1 = time.perf_counter()

    r = rep["BLOCK_MACRO_BEAM_SEARCH"]
    out = {
        "BLOCK_MACRO_BEAM_BENCH": {
            "seed": seed,
            "epsilon": epsilon,
            "depth": depth,
            "beam": beam,
            "macro_len": macro_len,
            "topM": topM,
            "H_macro": int(r["H_macro"]),
            "best_return_sum": float(r["best_return_sum"]),
            "best_risk_max": float(r["best_risk_max"]),
            "best_path_len": int(len(r["best_flat_path"])),
            "seconds": float(t1 - t0),
        }
    }
    out["BLOCK_MACRO_BEAM_SEARCH"] = search

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(
        "[PASS] BLOCK_MACRO_BEAM_BENCH_WRITTEN:"
        f" macro_len={macro_len}"
        f" topM={topM}"
        f" beam={beam}"
        f" depth={depth}"
        f" return_sum={float(r['best_return_sum']):.6f}"
        f" risk_max={float(r['best_risk_max']):.6f}"
        f" seconds={float(t1-t0):.6f}"
    )


if __name__ == "__main__":
    main()
