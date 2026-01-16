from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SpeakIntent:
    sid: Optional[int]
    tone: str
    strength: float
    goal: str
    must_include: List[str] = field(default_factory=list)
    must_not_include: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        return {
            "sid": self.sid,
            "tone": self.tone,
            "strength": float(self.strength),
            "goal": self.goal,
            "must_include": list(self.must_include),
            "must_not_include": list(self.must_not_include),
            "meta": dict(self.meta),
        }
