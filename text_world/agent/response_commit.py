from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


ResponseType = Literal[
    "EXECUTE_ACTION",
    "PROVIDE_INFORMATION",
    "ASK_CLARIFY",
    "SOCIAL_ACT",
    "ACTION_WITH_EXPLANATION",
]


@dataclass(frozen=True)
class ResponseCommit:
    r_type: ResponseType
    payload: Dict[str, Any]


@dataclass(frozen=True)
class GateParams:
    tau_accept: float = 0.95
    delta_gap: float = 0.10
    max_entropy_norm: float = 0.35


@dataclass(frozen=True)
class GateTelemetry:
    p_star: float
    p_2: float
    gap: float
    h_norm: float
    accept: bool
    reason: str
    fallback_r_type: Optional[str] = None
