from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class RenderOut:
    sid: int
    narration: str
    meta: Dict[str, Any]


def render_sid(sid: int) -> RenderOut:
    sid = int(sid)

    if sid == 0:
        return RenderOut(
            sid=sid,
            narration="You are in a room. The door is closed. You can open the door or look around.",
            meta={"door": "closed", "key_visible": True},
        )

    if sid == 16:
        return RenderOut(
            sid=sid,
            narration="The door is open. You can leave, or look around.",
            meta={"door": "open", "key_visible": True},
        )

    if sid == 80:
        return RenderOut(
            sid=sid,
            narration="You look around. The door is open. You see a key on the table.",
            meta={"door": "open", "key_visible": True},
        )

    return RenderOut(
        sid=sid,
        narration=f"State {sid} is active.",
        meta={"door": "unknown", "key_visible": None},
    )

def render_state(sid: int) -> RenderOut:
    return render_sid(int(sid))
