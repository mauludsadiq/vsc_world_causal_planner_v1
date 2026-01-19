from __future__ import annotations

from text_world.agent.debug_policy import emit_pass

import itertools
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from text_world.env_block import build_block_world, sample_transition
from text_world.render_block_clean import render_block_clean


@dataclass
class Node:
    s: int
    macro_path: List[List[int]]
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


def _det_step(seed: int, t: int, i: int, a: int, j: int) -> random.Random:
    return random.Random((seed + 1) * 1000003 + (t + 1) * 10007 + (i + 1) * 101 + int(a) * 17 + (j + 1) * 131)


def _exec_macro(world, s: int, macro: List[int], seed: int, t_macro: int, i_node: int) -> Tuple[int, float, float, List[int]]:
    s_cur = int(s)
    ret = 0.0
    risk_max = 0.0
    a_seq: List[int] = []
    for j, a in enumerate(macro):
        rng = _det_step(seed, t_macro, i_node, int(a), j)
        sp = sample_transition(world, s_cur, int(a), rng)
        obj = world.states[int(sp)]
        dr = _state_return(obj)
        rk = _state_risk(obj)
        ret += float(dr)
        risk_max = float(max(risk_max, rk))
        a_seq.append(int(a))
        s_cur = int(sp)
    return int(s_cur), float(ret), float(risk_max), a_seq


def _topM_actions(world, s: int, seed: int, t_macro: int, M: int) -> List[int]:
    scored: List[Tuple[float, int]] = []
    for a in world.actions:
        rng = _det_step(seed, t_macro, 0, int(a), 0)
        sp = sample_transition(world, int(s), int(a), rng)
        dr = _state_return(world.states[int(sp)])
        scored.append((float(dr), int(a)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[: max(1, int(M))]]


def run_block_macro_beam_search(
    out_json: str,
    seed: int = 0,
    epsilon: float = 0.15,
    depth: int = 40,
    beam: int = 16,
    macro_len: int = 2,
    topM: int = 16,
) -> Dict[str, Any]:
    world = build_block_world(n=max(8, beam * 2))
    s0 = 0

    H_macro = int((int(depth) + int(macro_len) - 1) // int(macro_len))

    frontier: List[Node] = [Node(s=int(s0), macro_path=[], ret_sum=0.0, risk_max=0.0)]
    rejected_counterfactuals: List[Dict[str, Any]] = []
    n_candidates_total = 0
    n_rejected_total = 0
    n_kept_total = 0

    for t_macro in range(H_macro):
        expanded: List[Node] = []
        for i, node in enumerate(frontier):
            top = _topM_actions(world, node.s, seed, t_macro, int(topM))
            macros = list(itertools.product(top, repeat=int(macro_len)))

            for macro in macros:
                n_candidates_total += 1
                s_out, dret, rk, a_seq = _exec_macro(world, node.s, list(macro), seed, t_macro, i)
                new_risk = float(max(node.risk_max, rk))
                new_ret = float(node.ret_sum + dret)

                if new_risk > float(epsilon):
                    n_rejected_total += 1
                    rejected_counterfactuals.append(
                        {
                            "t_macro": int(t_macro),
                            "from_state": int(node.s),
                            "macro": [int(x) for x in a_seq],
                            "epsilon": float(epsilon),
                            "risk_max_if_taken": float(new_risk),
                            "delta_return": float(dret),
                            "reason": "risk_budget_exceeded",
                        }
                    )
                    continue

                n_kept_total += 1
                expanded.append(
                    Node(
                        s=int(s_out),
                        macro_path=node.macro_path + [[int(x) for x in a_seq]],
                        ret_sum=float(new_ret),
                        risk_max=float(new_risk),
                    )
                )

        pool = expanded if expanded else frontier
        pool.sort(key=lambda n: n.ret_sum, reverse=True)
        frontier = pool[: max(1, int(beam))]

    best = frontier[0]
    flat_actions: List[int] = []
    for m in best.macro_path:
        flat_actions.extend(m)

    best_obj = world.states[int(best.s)]
    best_text = render_block_clean(best_obj)

    report = {
        "BLOCK_MACRO_BEAM_SEARCH": {
            "seed": int(seed),
            "epsilon": float(epsilon),
            "depth": int(depth),
            "beam": int(beam),
            "macro_len": int(macro_len),
            "topM": int(topM),
            "H_macro": int(H_macro),
            "best_macro_path": best.macro_path,
            "best_flat_path": flat_actions[: int(depth)],
            "best_return_sum": float(best.ret_sum),
            "best_risk_max": float(best.risk_max),
            "best_text": best_text,
            "n_candidates_total": int(n_candidates_total),
            "n_rejected_total": int(n_rejected_total),
            "n_kept_total": int(n_kept_total),
            "rejected_counterfactuals": rejected_counterfactuals[: 64],
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    emit_pass("[PASS] BLOCK_MACRO_BEAM_SEARCH_WRITTEN")
    emit_pass(
        "[PASS] BLOCK_MACRO_BEAM_SEARCH_BEST:"
        f" macro_len={macro_len}"
        f" topM={topM}"
        f" H_macro={H_macro}"
        f" return_sum={best.ret_sum:.4f} candidates={n_candidates_total} kept={n_kept_total} rejected={n_rejected_total}"
        f" risk_max={best.risk_max:.4f}<=eps={epsilon}"
    )

    return report
