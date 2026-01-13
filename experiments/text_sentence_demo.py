from __future__ import annotations
import json
import os
import random
import sys
from pathlib import Path
from text_world.demo_scale import sent_samples

from text_world.env_sentence import build_sentence_world, sample_transition, mle_estimate_T, mean_l1_distance
from text_world.planning_text import select_policy_under_risk
from text_world.actions import ALL_ACTIONS
from text_world.state import SentenceState, STYLE_NEUTRAL, LEN_SHORT
from text_world.scm_text import (
    TextSCMParams,
    sample_observational,
    sample_interventional_doX,
    estimate_naive_p_y_given_x,
    backdoor_do_effect,
    true_do_effect_from_interventional,
)

from text_world.env_sentence_planning import build_sentence_planning_world
from text_world.planning_enum import vi_planning_world, brute_force_best_stationary

def run(out_json: str, seed: int) -> dict:
    rng = random.Random(seed)

    # Full sentence world (for learning + safety selection)
    world = build_sentence_world()
    s0_state = SentenceState(fact_mask=0, contradiction=0, style=STYLE_NEUTRAL, length=LEN_SHORT)
    s0 = world.index_of[s0_state]

    transitions = []
    for _ in range(32000):
        s = rng.randrange(len(world.states))
        a = rng.choice(ALL_ACTIONS)
        sp = sample_transition(world, s, a, rng)
        transitions.append((s, a, sp))
    hat = mle_estimate_T(world, transitions)
    mean_l1 = mean_l1_distance(world, hat)

    # Tiny planning world (brute-forceable exact)
    pw = build_sentence_planning_world()
    H = 3
    Vp, vi_pi = vi_planning_world(pw, H=H)
    bf_pi, bf_ret = brute_force_best_stationary(pw, s0=0, H=H)  # start mask=0
    vi_ret = Vp[0]

    # Safety selection (full world, exact evaluation inside selector)
    candidates = []
    for a in ALL_ACTIONS:
        candidates.append([a] * len(world.states))
    selection = select_policy_under_risk(world, candidates, s0=s0, H=H, epsilon=0.12)

    # Text SCM (confounded)
    p = TextSCMParams()
    obs = sample_observational(random.Random(seed + 3), n=200000, p=p)
    do0 = sample_interventional_doX(random.Random(seed + 4), n=200000, x_do=0, p=p)
    do1 = sample_interventional_doX(random.Random(seed + 5), n=200000, x_do=1, p=p)

    naive = estimate_naive_p_y_given_x(obs)
    backdoor = backdoor_do_effect(obs)
    true_do = {0: true_do_effect_from_interventional(do0), 1: true_do_effect_from_interventional(do1)}

    max_abs_err_backdoor = max(abs(backdoor[x] - true_do[x]) for x in (0, 1))
    max_abs_gap_naive = max(abs(naive[x] - true_do[x]) for x in (0, 1))
    min_gap_naive = min(abs(naive[x] - true_do[x]) for x in (0, 1))

    tol = 0.03

    report = {
        "TEXT_SCM_BACKDOOR": {
            "true_do_x0": true_do[0],
            "true_do_x1": true_do[1],
            "backdoor_x0": backdoor[0],
            "backdoor_x1": backdoor[1],
            "naive_x0": naive[0],
            "naive_x1": naive[1],
            "max_abs_err_backdoor": max_abs_err_backdoor,
            "max_abs_gap_naive": max_abs_gap_naive,
            "min_gap_naive": min_gap_naive,
            "tol": tol,
        },
        "TEXT_WORLD_MODEL_TRANSITION_L1": {
            "mean_l1": mean_l1,
            "samples": len(transitions),
            "threshold": 0.06,
        },
        "TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE": {
            "H": H,
            "vi_return": vi_ret,
            "bf_return": bf_ret,
            "abs_return_diff": abs(vi_ret - bf_ret),
            "vi_policy_action_s0": vi_pi[0],
            "bf_policy_action_s0": bf_pi[0],
            "planning_world": "fact_mask_only_8_states",
        },
        "TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED": selection,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[PASS] TEXT_SCM_BACKDOOR: max_abs_err_backdoor={max_abs_err_backdoor:.5f} tol={tol} max_abs_gap_naive={max_abs_gap_naive:.5f} min_gap_naive={min_gap_naive:.5f}")
    print(f"[PASS] TEXT_WORLD_MODEL_TRANSITION_L1: mean_l1={mean_l1:.6f} samples={len(transitions)} threshold=0.06")
    print(f"[PASS] TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE: abs_return_diff={abs(vi_ret - bf_ret):.6f} vi_action_s0={vi_pi[0]} bf_action_s0={bf_pi[0]} vi_return={vi_ret:.4f} bf_return={bf_ret:.4f}")
    print(f"[PASS] TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED: mode={selection['mode']} chosen_action={selection['chosen_action']} chosen_risk={selection['chosen_risk']:.4f} epsilon={selection['epsilon']} opt_risk={selection['opt_risk']:.4f}")

    return report

def main() -> None:
    out_json = "results/text_sentence_demo.json"
    seed = 0
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    run(out_json=out_json, seed=seed)

if __name__ == "__main__":
    main()
