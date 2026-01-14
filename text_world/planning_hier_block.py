from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True, order=True)
class Macro:
    name: str
    args: Tuple[Any, ...]


def _stable_tag(seed: int, t_macro: int, s: int, macro: Macro) -> int:
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    h.update(str(t_macro).encode("utf-8"))
    h.update(b"|")
    h.update(str(s).encode("utf-8"))
    h.update(b"|")
    h.update(macro.name.encode("utf-8"))
    h.update(b"|")
    h.update(repr(macro.args).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big") % 2000000000


def enumerate_macros_from_primitives(world: Any, s: int) -> List[Macro]:
    actions = list(world.enumerate_actions(s))
    actions = [int(a) for a in actions]
    actions.sort()
    return [Macro("PRIM", (a,)) for a in actions]


def micro_eval_prim(world: Any, s0: int, macro: Macro, *, seed: int, t_macro: int) -> Dict[str, Any]:
    if macro.name != "PRIM" or len(macro.args) != 1:
        raise ValueError("expected Macro(name='PRIM', args=(a,))")
    a = int(macro.args[0])
    tag = _stable_tag(int(seed), int(t_macro), int(s0), macro)
    rng = random.Random(tag)
    sp, dr, rk = world.step(int(s0), a, rng)
    return {
        "s_out": int(sp),
        "a_seq": [a],
        "delta_return": float(dr),
        "risk_max": float(rk),
    }


def hier_beam_search_block(
    world: Any,
    s0: int,
    *,
    seed: int,
    beam: int,
    H_macro: int,
    epsilon: float,
) -> Dict[str, Any]:
    frontier: List[Dict[str, Any]] = [
        {"s": int(s0), "macro_trace": [], "micro_traces": [], "ret_sum": 0.0, "risk_max": 0.0, "audit": []}
    ]

    step_rejected: List[Dict[str, Any]] = []

    for t_macro in range(int(H_macro)):
        expanded: List[Dict[str, Any]] = []

        for node in frontier:
            s = int(node["s"])
            macros = enumerate_macros_from_primitives(world, s)

            scored: List[Tuple[Macro, Dict[str, Any]]] = []
            for macro in macros:
                micro = micro_eval_prim(world, s, macro, seed=int(seed), t_macro=int(t_macro))
                scored.append((macro, micro))

            best_rejected = None
            best_rejected_ret = None
            for macro, micro in scored:
                new_risk = float(max(float(node["risk_max"]), float(micro["risk_max"])))
                if new_risk > float(epsilon):
                    if best_rejected is None or float(micro["delta_return"]) > float(best_rejected_ret):
                        best_rejected = {
                            "t_macro": int(t_macro),
                            "from_state": int(s),
                            "macro": {"name": macro.name, "args": list(macro.args)},
                            "epsilon": float(epsilon),
                            "risk_max_if_taken": float(new_risk),
                            "delta_return": float(micro["delta_return"]),
                            "reason": "risk_budget_exceeded",
                        }
                        best_rejected_ret = float(micro["delta_return"])

            if best_rejected is not None:
                step_rejected.append(best_rejected)

            for macro, micro in scored:
                new_node = {
                    "s": int(micro["s_out"]),
                    "macro_trace": node["macro_trace"] + [macro],
                    "micro_traces": node["micro_traces"] + [list(micro["a_seq"])],
                    "ret_sum": float(node["ret_sum"] + float(micro["delta_return"])),
                    "risk_max": float(max(float(node["risk_max"]), float(micro["risk_max"]))),
                    "audit": node["audit"]
                    + [
                        {
                            "t_macro": int(t_macro),
                            "from_state": int(s),
                            "macro": {"name": macro.name, "args": list(macro.args)},
                            "micro": micro,
                        }
                    ],
                }
                expanded.append(new_node)

        safe = [n for n in expanded if float(n["risk_max"]) <= float(epsilon)]
        pool = safe if safe else expanded
        pool.sort(key=lambda n: float(n["ret_sum"]), reverse=True)
        frontier = pool[: max(1, int(beam))]

    best = frontier[0]
    return {
        "HIER_BLOCK_PLAN": {
            "seed": int(seed),
            "epsilon": float(epsilon),
            "H_macro": int(H_macro),
            "beam": int(beam),
            "ret_sum": float(best["ret_sum"]),
            "risk_max": float(best["risk_max"]),
            "macro_trace": [{"name": m.name, "args": list(m.args)} for m in best["macro_trace"]],
            "micro_traces": best["micro_traces"],
            "audit": best["audit"],
            "rejected_counterfactuals": step_rejected,
        }
    }
