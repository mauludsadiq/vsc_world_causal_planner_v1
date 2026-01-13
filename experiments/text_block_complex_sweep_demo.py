from __future__ import annotations
import json
import sys
from pathlib import Path

from text_world.env_block_complex import ComplexCfg, build_block_world_complex, mean_l1_over_anchors


def main() -> None:
    out_json = "results/text_block_complex_sweep_demo.json"
    seed = 0
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])

    grid = [
        {"n": 64, "towers": 2, "fragile_prob": 0.10, "grippers": 2},
        {"n": 128, "towers": 3, "fragile_prob": 0.15, "grippers": 3},
    ]

    rows = []
    for g in grid:
        cfg = ComplexCfg(n=g["n"], towers=g["towers"], fragile_prob=g["fragile_prob"], grippers=g["grippers"])
        world = build_block_world_complex(cfg)

        n_states = len(world["states"])
        anchors = list(range(min(32, n_states)))
        reps = 10

        mean_l1 = mean_l1_over_anchors(world, anchors, reps, seed)
        rows.append(
            {
                "cfg": g,
                "n_states": n_states,
                "anchors": len(anchors),
                "reps_per_key": reps,
                "mean_l1": mean_l1,
            }
        )

    data = {"BLOCK_COMPLEX_SWEEP": {"seed": seed, "rows": rows}}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("[PASS] BLOCK_COMPLEX_SWEEP_WRITTEN: out=" + out_json)


if __name__ == "__main__":
    main()
