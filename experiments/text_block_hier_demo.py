import json
from pathlib import Path

from text_world.env_block import build_block_world
from text_world.block_world_adapter import BlockWorldAdapter
from text_world.planning_hier_block import hier_beam_search_block


def main():
    import sys

    if len(sys.argv) < 5:
        raise SystemExit("usage: python -m experiments.text_block_hier_demo <out_json> <seed> <n_block> <H_macro>")

    out_json = sys.argv[1]
    seed = int(sys.argv[2])
    n_block = int(sys.argv[3])
    H_macro = int(sys.argv[4])

    epsilon = 0.15
    beam = 16

    world_raw = build_block_world(n=int(n_block))
    world = BlockWorldAdapter(world_raw)

    s0 = 0
    out = hier_beam_search_block(world, s0, seed=seed, beam=beam, H_macro=H_macro, epsilon=epsilon)

    report = {
        "TEXT_BLOCK_HIER_DEMO": {
            "world_kind": "env_block.build_block_world",
            "seed": seed,
            "n_block": n_block,
            "H_macro": H_macro,
            "beam": beam,
            "epsilon": float(epsilon),
            "plan": out["HIER_BLOCK_PLAN"],
        }
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    plan = out["HIER_BLOCK_PLAN"]
    print(
        "[PASS] TEXT_BLOCK_HIER_DEMO_WRITTEN:"
        f" world=env_block.build_block_world"
        f" ret_sum={plan['ret_sum']:.6f}"
        f" risk_max={plan['risk_max']:.6f}"
        f" epsilon={epsilon:.6f}"
        f" H_macro={H_macro}"
        f" beam={beam}"
        f" n_block={n_block}"
    )


if __name__ == "__main__":
    main()
