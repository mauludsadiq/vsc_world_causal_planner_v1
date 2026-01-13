from __future__ import annotations
from dataclasses import dataclass
import random
from text_world.demo_scale import trials
from typing import Any, Dict, Tuple

from text_world.env_block import build_block_world, sample_transition

@dataclass(frozen=True)
class CfResult:
    action: int
    mean_return: float
    mean_risk: float

def block_risk_event(world, state_index: int) -> int:
    b = world.states[state_index]
    if b.kappa == 0:
        return 1
    styles = [p.s1.style for p in b.paras]
    if any(s != styles[0] for s in styles):
        return 1
    return 0

def rollout(world, s0: int, action: int, H: int, rng: random.Random) -> Tuple[float, float]:
    s = s0
    ret = 0.0
    risk = 0.0
    for _ in range(H):
        sp = sample_transition(world, s, action, rng)
        b = world.states[sp]
        styles = [p.s1.style for p in b.paras]
        diversity = 1.0 if len(set(styles)) > 1 else 0.0
        ret += float(b.kappa) + 1.5 * diversity
        risk += float(block_risk_event(world, sp))
        s = sp
    return ret, (risk / H)

def mc_eval(world, s0: int, action: int, H: int, seed: int, trials: int) -> CfResult:
    rng = random.Random(seed)
    r_sum = 0.0
    k_sum = 0.0
    for _ in range(trials):
        r, k = rollout(world, s0, action, H, rng)
        r_sum += r
        k_sum += k
    return CfResult(action=action, mean_return=r_sum / trials, mean_risk=k_sum / trials)

def explain_counterfactual_block(seed: int, n: int, chosen_action: int, alt_action: int, H: int = 3, trials: int = 400) -> Dict[str, Any]:
    world = build_block_world(n=n)
    s0 = 0
    chosen = mc_eval(world, s0, chosen_action, H, seed + 11, trials)
    alt = mc_eval(world, s0, alt_action, H, seed + 29, trials)
    direction = "lower" if alt.mean_risk < chosen.mean_risk else "higher" if alt.mean_risk > chosen.mean_risk else "equal"
    explanation = (
        f"If we had taken action {alt_action} instead of {chosen_action}, "
        f"the expected risk would have been {alt.mean_risk:.4f} instead of {chosen.mean_risk:.4f} "
        f"and the expected return would have been {alt.mean_return:.4f} instead of {chosen.mean_return:.4f}. "
        f"The risk is {direction} because action choices change the probability of entering hazard-tagged states "
        f"(kappa=0 or inconsistent paragraph styles within a block)."
    )
    return {
        "seed": seed,
        "n": n,
        "H": H,
        "trials": trials,
        "chosen": {"action": chosen.action, "mean_return": chosen.mean_return, "mean_risk": chosen.mean_risk},
        "alt": {"action": alt.action, "mean_return": alt.mean_return, "mean_risk": alt.mean_risk},
        "explanation": explanation,
    }
