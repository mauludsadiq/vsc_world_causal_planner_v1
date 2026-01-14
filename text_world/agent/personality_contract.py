from __future__ import annotations
import json
from typing import Any, Dict, Optional

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
        json.dump(report, f, indent=2)

    print("[PASS] PERSONALITY_CONTRACT_WRITTEN")
    print("MAIN_REPLY:")
    print(main_reply)
    print("SAFETY_VERDICT:")
    print(json.dumps(safety, indent=2))
    print("COUNTERFACTUAL_REJECTED_BRANCH:")
    print(json.dumps(rejected, indent=2))
    return report
