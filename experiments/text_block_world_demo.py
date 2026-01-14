from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

from text_world.env_block import build_block_world, sample_transition, mle_estimate_T, mean_l1_over_keys


def run(out_json: str, seed: int, n: int, anchors: int, reps: int, threshold: float) -> dict:
    rng = random.Random(int(seed))
    world = build_block_world(int(n))

    s0 = 0
    discovered = [s0]
    discovered_set = {s0}

    warm = max(2000, 200 * int(n))
    for _ in range(warm):
        s = discovered[rng.randrange(len(discovered))]
        act = world.actions[rng.randrange(len(world.actions))]
        sp = sample_transition(world, int(s), int(act), rng)
        if int(sp) not in discovered_set:
            discovered_set.add(int(sp))
            discovered.append(int(sp))

    a = min(int(anchors), len(discovered))
    anchor_states = rng.sample(discovered, a) if a > 0 else [0]

    transitions: List[Tuple[int, int, int]] = []
    for s in anchor_states:
        for _ in range(int(reps)):
            act = world.actions[rng.randrange(len(world.actions))]
            sp = sample_transition(world, int(s), int(act), rng)
            transitions.append((int(s), int(act), int(sp)))

    hat = mle_estimate_T(world, transitions)
    mean_l1 = mean_l1_over_keys(world, hat)
    samples = len(transitions)

    report = {
        "BLOCK_WORLD_MODEL_TRANSITION_L1": {
            "n": int(n),
            "seed": int(seed),
            "anchors": int(a),
            "reps": int(reps),
            "samples": int(samples),
            "threshold": float(threshold),
            "mean_l1": float(mean_l1),
        }
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "[PASS] BLOCK_WORLD_MODEL_TRANSITION_L1:"
        f" n={n}"
        f" mean_l1={mean_l1:.6f}"
        f" anchors={a}"
        f" reps={reps}"
        f" samples={samples}"
        f" threshold={threshold}"
    )

    if float(mean_l1) > float(threshold):
        raise SystemExit(f"mean_l1 {mean_l1} exceeds threshold {threshold}")

    return report


def main() -> None:
    out_json = "results/text_block_world_demo.json"
    seed = 0
    n = 8
    anchors = 32
    reps = 5
    threshold = 0.30

    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    if len(sys.argv) >= 4:
        n = int(sys.argv[3])
    if len(sys.argv) >= 5:
        anchors = int(sys.argv[4])
    if len(sys.argv) >= 6:
        reps = int(sys.argv[5])
    if len(sys.argv) >= 7:
        threshold = float(sys.argv[6])

    run(out_json, seed, n, anchors, reps, threshold)


if __name__ == "__main__":
    main()
