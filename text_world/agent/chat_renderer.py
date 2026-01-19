from __future__ import annotations
from typing import Any, Dict

from text_world.agent.response_commit import ResponseCommit


def render_user_text(commit: ResponseCommit, world_view: Dict[str, Any]) -> str:
    r_type = commit.r_type
    payload = commit.payload

    if r_type == "ASK_CLARIFY":
        q = payload.get("question") or "Can you clarify what you mean?"
        return str(q)

    if r_type == "SOCIAL_ACT":
        text = payload.get("text") or "Got it."
        return str(text)

    if r_type == "PROVIDE_INFORMATION":
        text = payload.get("text") or "Here’s what I can tell you."
        return str(text)

    if r_type == "EXECUTE_ACTION":
        action_text = payload.get("action_text") or "I did it."
        consequence = payload.get("consequence_text") or ""
        if consequence:
            return f"{action_text}\n{consequence}"
        return str(action_text)

    if r_type == "ACTION_WITH_EXPLANATION":
        action_text = payload.get("action_text") or "Done."
        explanation = payload.get("explanation_text") or ""
        if explanation:
            return f"{action_text}\n{explanation}"
        return str(action_text)

    return "Okay."
