from __future__ import annotations
import json
import random
import sys
from pathlib import Path

from text_world.env_document import build_document_world, sample_transition

def doc_hazard(d) -> int:
    return int(d.kappa == 0)

def style_mismatch_risk(d) -> int:
    styles = [d.p0.s1.style, d.p1.s1.style, d.p2.s1.style, d.p3.s1.style]
    return int(any(s != styles[0] for s in styles))

def rollout_return_and_risk(world, s0, policy, H, rng):
    s = s0
    ret = 0.0
    risk_events = 0.0
    for t in range(H):
        act = policy(t)
        sp = sample_transition(world, s, act, rng)
        d = world.states[sp]
        # return: coherence + small bonus for "formalness diversity" (creates incentive to mismatch)
        styles = [d.p0.s1.style, d.p1.s1.style, d.p2.s1.style, d.p3.s1.style]
        diversity = 1.0 if len(set(styles)) > 1 else 0.0
        ret += float(d.kappa) + 1.5 * diversity
        risk_events += float(doc_hazard(d) or style_mismatch_risk(d))
        s = sp
    return ret, (risk_events / H)

def mc_score(world, s0, policy, H, rng, trials=400):
    r_sum = 0.0
    k_sum = 0.0
    for _ in range(trials):
        r, k = rollout_return_and_risk(world, s0, policy, H, rng)
        r_sum += r
        k_sum += k
    return (r_sum / trials, k_sum / trials)

def run(out_json: str, seed: int) -> dict:
    rng = random.Random(seed)
    world = build_document_world()
    s0 = 0
    H = 3
    epsilon = 0.10

    # Candidate class A: constant actions (baseline)
    const_pols = []
    for act in world.actions:
        const_pols.append(("const", act, lambda t, a=act: a))

    # Candidate class B: "booster" policies that attempt to induce style mismatch early
    # We pick a small, finite set: first step edits block i with para_action b, then repeats a main action.
    # This stays brute-forceable: 4 blocks * 27 boosters * 27 main = 2916 policies.
    booster_para_actions = world.pw.actions[:]  # 27
    main_actions = world.actions[:]             # 108

    booster_pols = []
    for block in (0, 1, 2, 3):
        for b in booster_para_actions:
            booster_act = block * 64 + b
            for main in main_actions:
                def make_policy(boost=booster_act, main_act=main):
                    return lambda t: (boost if t == 0 else main_act)
                booster_pols.append(("boost", (booster_act, main), make_policy()))

    candidates = const_pols + booster_pols

    scored = []
    for kind, meta, pol in candidates:
        r, k = mc_score(world, s0, pol, H, rng, trials=250)
        scored.append((kind, meta, r, k))

    opt = max(scored, key=lambda x: x[2])
    feasible = [x for x in scored if x[3] <= epsilon]
    chosen = max(feasible, key=lambda x: x[2]) if feasible else min(scored, key=lambda x: x[3])

    mode = "feasible_best_return" if feasible else "fallback_min_risk"

    report = {
        "DOC_SAFETY_TRADEOFF_FORCED": {
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
        }
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    forced = (opt[3] > epsilon) and (chosen[1] != opt[1] or chosen[0] != opt[0])

    print(f"[PASS] DOC_SAFETY_TRADEOFF_FORCED: mode={mode} epsilon={epsilon} chosen_risk={chosen[3]:.4f} opt_risk={opt[3]:.4f} forced={forced}")
    if not forced:
        # If this ever becomes degenerate for a seed, fail loudly.
        raise SystemExit("tradeoff not forced for this seed; adjust reward/risk shaping")

    return report

def main() -> None:
    out_json = "results/text_document_tradeoff_demo.json"
    seed = 0
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    run(out_json, seed)

if __name__ == "__main__":
    main()
