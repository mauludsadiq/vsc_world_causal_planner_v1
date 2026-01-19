from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Tuple

from text_world.agent.response_commit import ResponseCommit, GateParams, GateTelemetry


def gate_commit(
    commit: ResponseCommit,
    parse_metrics: Dict[str, Any],
    params: GateParams,
    fallback: ResponseCommit,
) -> Tuple[ResponseCommit, GateTelemetry]:
    p_star = float(parse_metrics.get("best_prob", 0.0))
    p_2 = float(parse_metrics.get("second_prob", 0.0))
    h_norm = float(parse_metrics.get("H_norm", 1.0))
    gap = p_star - p_2

    accept = (p_star >= params.tau_accept) and (gap >= params.delta_gap) and (h_norm <= params.max_entropy_norm)

    if accept:
        tel = GateTelemetry(
            p_star=p_star,
            p_2=p_2,
            gap=gap,
            h_norm=h_norm,
            accept=True,
            reason="ACCEPT",
            fallback_r_type=None,
        )
        return commit, tel

    tel = GateTelemetry(
        p_star=p_star,
        p_2=p_2,
        gap=gap,
        h_norm=h_norm,
        accept=False,
        reason="REJECT_TO_FALLBACK",
        fallback_r_type=fallback.r_type,
    )
    return fallback, tel


def telemetry_to_dict(tel: GateTelemetry) -> Dict[str, Any]:
    return asdict(tel)
