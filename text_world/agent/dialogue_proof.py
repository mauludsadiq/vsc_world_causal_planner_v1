from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import hashlib
import json


@dataclass(frozen=True)
class DecodeProof:
    mode: str
    sid_hat: Optional[int]
    p_top1: Optional[float]
    p_top2: Optional[float]
    margin: Optional[float]
    entropy: Optional[float]
    tau_p: float
    tau_margin: float
    seed: int
    reason: Optional[str] = None


@dataclass(frozen=True)
class TurnProof:
    turn: int
    user_text: str
    decode: DecodeProof

    sid_in: int
    sid_out: Optional[int]

    assistant_text: Optional[str]

    main_reply: Optional[str]
    safety_verdict: Dict[str, Any]
    rejected_counterfactuals: List[Dict[str, Any]] = field(default_factory=list)
    main_reply: Optional[str]
    safety_verdict: Dict[str, Any]


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(obj: Any) -> str:
    b = _stable_json(obj).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def build_dialogue_proof(seed: int, turns: List[TurnProof]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "seed": int(seed),
        "turns": [asdict(t) for t in turns],
    }
    for td in out["turns"]:
        if "decode" in td and "decoded" not in td:
            td["decoded"] = td["decode"]
    out["sha256"] = sha256_hex(out)
    return out
