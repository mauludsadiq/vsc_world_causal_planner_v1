from __future__ import annotations

from text_world.agent.debug_policy import emit_user, emit_pass


import json
from typing import Any, Dict


def emit_personality_reply(
    main_reply: str,
    safety: Dict[str, Any],
    rejected: Dict[str, Any],
    out_json: str,
) -> Dict[str, Any]:
    report = {
        "PERSONALITY_CONTRACT": {
            "main_reply": main_reply,
            "safety_verdict": safety,
            "mandatory_counterfactual": rejected,
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    emit_pass("[PASS] PERSONALITY_CONTRACT_WRITTEN")

    emit_user(main_reply)
    return report
