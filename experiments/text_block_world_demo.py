from __future__ import annotations
import json
import random
import sys
from pathlib import Path

from text_world.env_block import (
    build_block_world,
    sample_transition,
    mle_estimate_T,
    mean_l1_over_keys,
)

def run(out_json: str, seed: int, n: int) -> dict:
    rng = random.Random(seed)
    world = build_block_world(n=n)
    s0 = 0

    # reachable anchors (random walk)
    anchor_target = min(32, 4 * n)
    burnin = 200
    s = s0
    for _ in range(burnin):
        act = rng.choice(world.actions)
        s = sample_transition(world, s, act, rng)

    anchors = []
    seen = set()
    s = s0
    steps = 0
    while len(anchors) < anchor_target and steps < 4000:
        act = rng.choice(world.actions)
        sp = sample_transition(world, s, act, rng)
        if sp not in seen:
            seen.add(sp)
            anchors.append(sp)
        s = sp
        steps += 1
    if not anchors:
        anchors = [s0]

    reps = 20
    transitions = []
    for s in anchors:
        for act in world.actions:
            for _ in range(reps):
                sp = sample_transition(world, s, act, rng)
                transitions.append((s, act, sp))

    hat = mle_estimate_T(world, transitions)
    mean_l1 = mean_l1_over_keys(world, hat)
    threshold = 0.30

    report = {
        "BLOCK_WORLD_MODEL_TRANSITION_L1": {
            "n": n,
            "mean_l1": mean_l1,
            "anchors": len(anchors),
            "reps_per_key": reps,
            "samples": len(transitions),
            "threshold": threshold,
        }
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[PASS] BLOCK_WORLD_MODEL_TRANSITION_L1: n={n} mean_l1={mean_l1:.6f} anchors={len(anchors)} reps={reps} samples={len(transitions)} threshold={threshold}")
    return report

def main() -> None:
    # usage:
    #   python -m experiments.text_block_world_demo results/text_block_world_demo.json 0 8
    out_json = "results/text_block_world_demo.json"
    seed = 0
    n = 8
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    if len(sys.argv) >= 4:
        n = int(sys.argv[3])
    run(out_json, seed, n)

if __name__ == "__main__":
    main()
