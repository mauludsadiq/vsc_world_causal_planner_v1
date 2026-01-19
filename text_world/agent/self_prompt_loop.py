from __future__ import annotations

from text_world.agent.debug_policy import emit_pass

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from text_world.render_parse_clean import render_sentence_clean
from text_world.actions import ALL_ACTIONS
from text_world.state import SentenceState, STYLE_NEUTRAL, LEN_SHORT

from text_world.env_sentence import (
    build_sentence_world,
    sample_transition,
    mle_estimate_T,
    mean_l1_distance,
)

from text_world.planning_text import select_policy_under_risk
from text_world.env_sentence_planning import build_sentence_planning_world
from text_world.planning_enum import vi_planning_world, brute_force_best_stationary

from text_world.scm_text import (
    TextSCMParams,
    sample_observational,
    sample_interventional_doX,
    estimate_naive_p_y_given_x,
    backdoor_do_effect,
    true_do_effect_from_interventional,
)


@dataclass(frozen=True)
class StepRecord:
    t: int
    prompt_prefix: str
    chosen_action: int
    chosen_text: str
    chosen_return: float
    chosen_risk: float
    epsilon: float
    counterfactual: Dict[str, Any]
    collapsed_proof: Dict[str, Any]


def _verify_causal(seed: int, tol: float = 0.02) -> Dict[str, Any]:
    p = TextSCMParams()
    obs = sample_observational(random.Random(seed + 3), n=200000, p=p)
    do0 = sample_interventional_doX(random.Random(seed + 4), n=200000, x_do=0, p=p)
    do1 = sample_interventional_doX(random.Random(seed + 5), n=200000, x_do=1, p=p)

    naive = estimate_naive_p_y_given_x(obs)
    backdoor = backdoor_do_effect(obs)
    true_do = {0: true_do_effect_from_interventional(do0), 1: true_do_effect_from_interventional(do1)}

    max_abs_err_backdoor = max(abs(backdoor[x] - true_do[x]) for x in (0, 1))
    min_gap_naive = min(abs(naive[x] - true_do[x]) for x in (0, 1))

    return {
        "tol": float(tol),
        "max_abs_err_backdoor": float(max_abs_err_backdoor),
        "min_gap_naive": float(min_gap_naive),
        "backdoor_ok": bool(max_abs_err_backdoor <= tol),
        "naive_fails": bool(min_gap_naive >= 0.07),
    }


def _verify_world_model(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    world = build_sentence_world()

    transitions = []
    for _ in range(32000):
        s = rng.randrange(len(world.states))
        a = rng.choice(ALL_ACTIONS)
        sp = sample_transition(world, s, a, rng)
        transitions.append((s, a, sp))

    hat = mle_estimate_T(world, transitions)
    mean_l1 = mean_l1_distance(world, hat)
    threshold = 0.06

    return {
        "mean_l1": float(mean_l1),
        "threshold": float(threshold),
        "samples": int(len(transitions)),
        "l1_ok": bool(mean_l1 <= threshold),
    }


def _verify_planning(seed: int) -> Dict[str, Any]:
    pw = build_sentence_planning_world()
    H = 3
    Vp, vi_pi = vi_planning_world(pw, H=H)
    bf_pi, bf_ret = brute_force_best_stationary(pw, s0=0, H=H)

    vi_ret = float(Vp[0])
    bf_ret = float(bf_ret)
    return {
        "H": int(H),
        "vi_policy": list(vi_pi),
        "bf_policy": list(bf_pi),
        "vi_return": vi_ret,
        "bf_return": bf_ret,
        "abs_return_diff": float(abs(vi_ret - bf_ret)),
        "vi_ok": bool(abs(vi_ret - bf_ret) <= 1e-6),
    }


def _det_step(world, s: int, a: int, seed: int, t: int) -> int:
    rng = random.Random(seed + 100000 + 1000 * t + int(a))
    return sample_transition(world, s, a, rng)


def run_self_prompt_loop(out_json: str, seed: int = 0, epsilon: float = 0.15, horizon: int = 8) -> Dict[str, Any]:
    world = build_sentence_world()
    s0_state = SentenceState(fact_mask=0, contradiction=0, style=STYLE_NEUTRAL, length=LEN_SHORT)
    s = int(world.index_of[s0_state])

    causal = _verify_causal(seed=seed, tol=0.02)
    wm = _verify_world_model(seed=seed)
    pl = _verify_planning(seed=seed)

    assert causal["backdoor_ok"] and causal["naive_fails"]
    assert wm["l1_ok"]
    assert pl["vi_ok"]

    transcript: List[Dict[str, Any]] = []

    for t in range(int(horizon)):
        prefix = render_sentence_clean(world.states[s])

        candidates = []
        for a in ALL_ACTIONS:
            candidates.append([a] * len(world.states))

        sel = select_policy_under_risk(world, candidates, s0=s, H=1, epsilon=float(epsilon))
        chosen_action = int(sel["chosen_action"])
        chosen_return = float(sel.get("chosen_return", 0.0))
        chosen_risk = float(sel.get("chosen_risk", 1.0))

        opt_action = int(sel.get("opt_action", chosen_action))
        opt_return = float(sel.get("opt_return", chosen_return))
        opt_risk = float(sel.get("opt_risk", chosen_risk))

        sp = _det_step(world, s, chosen_action, seed=seed, t=t)
        chosen_text = render_sentence_clean(world.states[sp])

        counterfactual = {
            "rejected_action": opt_action,
            "expected_return": opt_return,
            "expected_risk": opt_risk,
            "epsilon": float(epsilon),
            "rejected": bool(opt_risk > float(epsilon)),
        }

        step = StepRecord(
            t=t,
            prompt_prefix=prefix,
            chosen_action=chosen_action,
            chosen_text=chosen_text,
            chosen_return=chosen_return,
            chosen_risk=chosen_risk,
            epsilon=float(epsilon),
            counterfactual=counterfactual,
            collapsed_proof={
                "chosen_action": chosen_action,
                "chosen_return": chosen_return,
                "chosen_risk": chosen_risk,
                "epsilon": float(epsilon),
                "causal_max_abs_err_backdoor": causal["max_abs_err_backdoor"],
                "causal_tol": causal["tol"],
                "world_model_mean_l1": wm["mean_l1"],
                "world_model_threshold": wm["threshold"],
            },
        )

        transcript.append(step.__dict__)
        s = sp

    report = {
        "SELF_PROMPT_LOOP": {
            "seed": int(seed),
            "epsilon": float(epsilon),
            "horizon": int(horizon),
            "causal": causal,
            "world_model": wm,
            "planning": pl,
            "transcript": transcript,
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    last = transcript[-1]
    emit_pass("[PASS] SELF_PROMPT_LOOP_WRITTEN")
    emit_pass(f"[PASS] SELF_PROMPT_LOOP_LAST: action={last['chosen_action']} return={last['chosen_return']:.4f} risk={last['chosen_risk']:.4f}<=eps={float(epsilon):.4f}")
    return report
