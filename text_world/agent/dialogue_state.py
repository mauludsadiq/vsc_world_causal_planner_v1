from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DialogueTurn:
    turn: int
    user_text: str
    decode: Dict[str, Any]
    sid_out: int
    main_reply: str
    safety_verdict: Dict[str, Any]
    rejected_counterfactuals: List[Dict[str, Any]]


@dataclass(frozen=True)
class DialogueTrace:
    seed: int
    epsilon: float
    model_dir: Optional[str]
    tau_p: float
    tau_margin: float
    turns: List[Dict[str, Any]]


def new_trace(*, seed: int, epsilon: float, model_dir: Optional[str], tau_p: float, tau_margin: float) -> Dict[str, Any]:
    return asdict(
        DialogueTrace(
            seed=int(seed),
            epsilon=float(epsilon),
            model_dir=model_dir,
            tau_p=float(tau_p),
            tau_margin=float(tau_margin),
            turns=[],
        )
    )


def append_turn(trace: Dict[str, Any], turn: DialogueTurn) -> None:
    trace["turns"].append(asdict(turn))
