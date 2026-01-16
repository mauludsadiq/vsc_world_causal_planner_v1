from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class DialogueTurn:
    t: int
    user_text: str
    sid_in: Optional[int]
    sid_out: Optional[int]
    main_reply: str
    chosen_action: Optional[int]
    risk_max: Optional[float]
    epsilon: Optional[float]


@dataclass
class DialogueState:
    seed: int
    epsilon: float
    turn: int = 0
    last_sid: Optional[int] = None
    goal: Optional[str] = None
    history: List[DialogueTurn] = field(default_factory=list)

    def next_turn(self) -> int:
        self.turn += 1
        return self.turn

    def set_goal_if_missing(self, user_text: str) -> None:
        if self.goal is None and user_text.strip():
            self.goal = user_text.strip()

    def record(
        self,
        user_text: str,
        sid_in: Optional[int],
        sid_out: Optional[int],
        main_reply: str,
        chosen_action: Optional[int],
        risk_max: Optional[float],
    ) -> None:
        self.last_sid = sid_out if sid_out is not None else self.last_sid
        self.history.append(
            DialogueTurn(
                t=self.turn,
                user_text=user_text,
                sid_in=sid_in,
                sid_out=sid_out,
                main_reply=main_reply,
                chosen_action=chosen_action,
                risk_max=risk_max,
                epsilon=self.epsilon,
            )
        )

    def short_summary(self) -> str:
        g = self.goal if self.goal is not None else ""
        return f"turn={self.turn} goal={g} last_sid={self.last_sid} epsilon={self.epsilon}"
