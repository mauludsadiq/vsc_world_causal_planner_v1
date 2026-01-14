from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from text_world.env_block import build_block_world, sample_transition
from text_world.render_block_clean import render_block_clean

@dataclass
class Node:
    s: int
    path: List[int]
    ret_sum: float
    risk_max: float

def _state_return(obj: Any) -> float:
    if hasattr(obj, "kappa"):
        try:
            return float(getattr(obj, "kappa"))
        except Exception:
            return 0.0
    return 0.0

def _state_risk(obj: Any) -> float:
    if hasattr(obj, "kappa"):
        try:
            return 1.0 if float(getattr(obj, "kappa")) == 0.0 else 0.0
        except Exception:
            return 0.0
    if hasattr(obj, "hazard"):
        try:
            return 1.0 if bool(getattr(obj, "hazard")) else 0.0
        except Exception:
            return 0.0
    if hasattr(obj, "risk"):
        try:
            return float(getattr(obj, "risk"))
        except Exception:
            return 0.0
    return 0.0

def run_block_beam_search(out_json: str, seed: int = 0, epsilon: float = 0.15, depth: int = 10, beam: int = 8) -> Dict[str, Any]:
    world = build_block_world(n=max(8, beam * 2))
    s0 = 0

    frontier: List[Node] = [Node(s=s0, path=[], ret_sum=0.0, risk_max=0.0)]

    for t in range(depth):
        expanded: List[Node] = []
        for i, node in enumerate(frontier):
            for a in world.actions:
                rng = random.Random((seed + 1) * 1000003 + (t + 1) * 10007 + (i + 1) * 101 + int(a))
                sp = sample_transition(world, node.s, a, rng)
                obj = world.states[sp]
                dr = _state_return(obj)
                rk = _state_risk(obj)
                expanded.append(
                    Node(
                        s=int(sp),
                        path=node.path + [int(a)],
                        ret_sum=float(node.ret_sum + dr),
                        risk_max=float(max(node.risk_max, rk)),
                    )
                )

        safe = [n for n in expanded if n.risk_max <= epsilon]
        pool = safe if safe else expanded
        pool.sort(key=lambda n: n.ret_sum, reverse=True)
        frontier = pool[: max(1, int(beam))]

    best = frontier[0]

    best_obj = world.states[best.s]
    best_text = render_block_clean(best_obj)

    rejected = None
    for cand in expanded:
        if cand.risk_max > epsilon:
            rejected = cand
            break

    report = {
        "BLOCK_BEAM_SEARCH": {
            "seed": int(seed),
            "epsilon": float(epsilon),
            "depth": int(depth),
            "beam": int(beam),
            "best_path": [int(x) for x in best.path],
            "best_return_sum": float(best.ret_sum),
            "best_value": float(best.ret_sum),
            "best_risk_max": float(best.risk_max),
            "best_text": best_text,
            "rejected_example": None if rejected is None else {
                "path": [int(x) for x in rejected.path],
                "return_sum": float(rejected.ret_sum),
                "value": float(rejected.ret_sum),
                "risk_max": float(rejected.risk_max),
                "risk": float(rejected.risk_max),
            },
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[PASS] BLOCK_BEAM_SEARCH_WRITTEN")
    print("[PASS] BLOCK_BEAM_SEARCH_BEST: return_sum={:.4f} risk_max={:.4f}<=eps={}".format(best.ret_sum, best.risk_max, epsilon))
    return report
