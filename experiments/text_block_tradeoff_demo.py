from __future__ import annotations
import json
import random
import sys
from pathlib import Path

from text_world.env_block import build_block_world, sample_transition

def block_risk_events(b) -> int:
    if b.kappa == 0:
        return 1
    styles = [p.s1.style for p in b.paras]
    if any(s != styles[0] for s in styles):
        return 1
    return 0

def rollout_return_and_risk(world, s0, act, H, rng):
    s = s0
    ret = 0.0
    risk = 0.0
    for _ in range(H):
        sp = sample_transition(world, s, act, rng)
        b = world.states[sp]
        styles = [p.s1.style for p in b.paras]
        diversity = 1.0 if len(set(styles)) > 1 else 0.0
        ret += float(b.kappa) + 1.5 * diversity
        risk += float(block_risk_events(b))
        s = sp
    return ret, (risk / H)

def mc_score_const(world, s0, act, H, rng, trials=200):
    r_sum = 0.0
    k_sum = 0.0
    for _ in range(trials):
        r, k = rollout_return_and_risk(world, s0, act, H, rng)
        r_sum += r
        k_sum += k
    return (r_sum / trials, k_sum / trials)

def run(out_json: str, seed: int, n: int) -> dict:
    rng = random.Random(seed)
    world = build_block_world(n=n)
    s0 = 0
    H = 3
    epsilon = 0.10

    # CONST-ONLY: evaluate each constant action once (finite: n*27)
    scored = []
    for act in world.actions:
        r, k = mc_score_const(world, s0, act, H, rng, trials=200)
        scored.append(("const", act, r, k))

    opt = max(scored, key=lambda x: x[2])
    feasible = [x for x in scored if x[3] <= epsilon]
    chosen = max(feasible, key=lambda x: x[2]) if feasible else min(scored, key=lambda x: x[3])
    mode = "feasible_best_return" if feasible else "fallback_min_risk"

    forced = (opt[3] > epsilon) and (chosen[1] != opt[1] or chosen[0] != opt[0])

    report = {
        "BLOCK_SAFETY_TRADEOFF_FORCED": {
            "n": n,
            "mode": mode,
            "epsilon": epsilon,
            "chosen_kind": chosen[0],
            "chosen_meta": chosen[1],
            "chosen_return": chosen[2],
            "chosen_risk": chosen[3],
            "opt_kind": opt[0],
            "opt_meta": opt[1],
            "opt_return": opt[2],
            "opt_risk": opt[3],
            "forced": forced,
        }
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[PASS] BLOCK_SAFETY_TRADEOFF_FORCED: n={n} mode={mode} epsilon={epsilon} chosen_risk={chosen[3]:.4f} opt_risk={opt[3]:.4f} forced={forced}")
    if not forced:
        raise SystemExit("tradeoff not forced for this seed/n; adjust shaping or epsilon")

    return report

def main() -> None:
    out_json = "results/text_block_tradeoff_demo.json"
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
