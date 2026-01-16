from __future__ import annotations

import json
import os
import sys
from typing import Optional, Tuple

from text_world.agent.dialogue_state import DialogueState
from text_world.agent.intent_schema import SpeakIntent
from text_world.agent.speaker_surface import SpeakerSurface
from text_world.agent.planner_bridge import step_safe


def _try_symbolic_parse(text: str) -> Optional[int]:
    s = text.strip()
    if s.isdigit():
        return int(s)

    try:
        from text_world.render_parse_clean import parse_sentence_clean
        st = parse_sentence_clean(text)
        if hasattr(st, "sid"):
            return int(getattr(st, "sid"))
        if hasattr(st, "to_id"):
            return int(st.to_id())
        return None
    except Exception:
        return None


def _render_from_sid(sid: Optional[int]) -> str:
    if sid is None:
        return "the claim is unspecified"
    try:
        from text_world.enhanced_sentence import enhanced_render
        return enhanced_render(int(sid), strength=0.65, sugar=True)
    except Exception:
        return f"state {int(sid)} is active"


def _select_next_sid(sid_in: Optional[int], epsilon: float, seed: int) -> Tuple[Optional[int], Optional[int], float, str, dict]:
    if sid_in is None:
        return None, None, 0.0, "no_state", {"reason": "no sid parsed"}
    out = step_safe(int(sid_in), float(epsilon), int(seed))
    return int(out.sid_out), int(out.chosen_action), float(out.chosen_risk), str(out.mode), dict(out.rejected)


def main() -> int:
    if len(sys.argv) < 3:
        print("USAGE: python -m text_world.agent.chat_loop SEED EPSILON", file=sys.stderr)
        return 2

    seed = int(sys.argv[1])
    epsilon = float(sys.argv[2])

    ds = DialogueState(seed=seed, epsilon=epsilon)
    speaker = SpeakerSurface(seed=seed)

    print("CHAT_LOOP_READY")
    print(json.dumps({"seed": seed, "epsilon": epsilon}, indent=2))

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break

        ds.next_turn()
        ds.set_goal_if_missing(line)

        sid_in = _try_symbolic_parse(line)
        sid_out, chosen_action, risk_max, mode, rejected = _select_next_sid(sid_in, ds.epsilon, seed)

        base = _render_from_sid(sid_out)

        intent = SpeakIntent(
            sid=sid_out,
            tone=os.environ.get("TONE", "neutral"),
            strength=float(os.environ.get("STRENGTH", "0.65")),
            goal=ds.goal or "",
            must_include=["sid_out", "epsilon", "risk_max"],
            must_not_include=["unsafe_instructions"],
            meta={"turn": ds.turn, "sid_in": sid_in, "chosen_action": chosen_action, "risk_max": risk_max},
        )

        main_reply = speaker.speak(base_text=base, intent=intent, turn=ds.turn)

        ds.record(
            user_text=line,
            sid_in=sid_in,
            sid_out=sid_out,
            main_reply=main_reply,
            chosen_action=chosen_action,
            risk_max=risk_max,
        )

        safety = {"mode": mode, "epsilon": ds.epsilon, "risk_max": risk_max, "ok": bool(risk_max <= ds.epsilon), "chosen_action": chosen_action, "sid_in": sid_in, "sid_out": sid_out}
        rejected = rejected

        print("MAIN_REPLY:")
        print(main_reply)
        print("SAFETY_VERDICT:")
        print(json.dumps(safety, indent=2))
        print("COUNTERFACTUAL_REJECTED_BRANCH:")
        print(json.dumps(rejected, indent=2))

    print("CHAT_LOOP_DONE")
    print(ds.short_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
