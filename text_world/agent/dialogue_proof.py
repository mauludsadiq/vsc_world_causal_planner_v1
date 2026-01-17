from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import hashlib
import json


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return sha256_bytes(stable_json_dumps(obj).encode("utf-8"))


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
    decoded: DecodeProof
    state_id_in: int
    state_id_out: Optional[int]
    assistant_text: Optional[str]


@dataclass(frozen=True)
class DialogueProof:
    seed: int
    n_turns: int
    turns: List[TurnProof]
    sha256: str


def build_dialogue_proof(seed: int, turns: List[TurnProof]) -> Dict[str, Any]:
    payload = {
        "seed": int(seed),
        "n_turns": int(len(turns)),
        "turns": [asdict(t) for t in turns],
    }
    payload["sha256"] = sha256_json(payload)
    return payload
