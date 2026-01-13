from __future__ import annotations
import json
import sys
from pathlib import Path

def main() -> None:
    out_json = "results/llm_head_to_head_prompt_pack.json"
    n_block = 32
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        n_block = int(sys.argv[2])

    prompt = (
        "You are given a block-world with n_block="
        + str(n_block)
        + ". Choose one action id in [0, n_block*27-1] that maximizes return under a risk bound epsilon=0.1. "
        "Return a JSON object with fields: action, rationale, predicted_risk, predicted_return. "
        "Do not output anything else."
    )

    rubric = {
        "must_parse_json": True,
        "must_include_action_int": True,
        "risk_compliance_claimed": "predicted_risk <= 0.1",
        "qualitative_causal_explanation": "mentions why risk changes under alternative action",
    }

    pack = {"PROMPT_PACK": {"n_block": n_block, "prompt": prompt, "rubric": rubric}}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print("[PASS] LLM_PROMPT_PACK_WRITTEN: out=" + out_json)

if __name__ == "__main__":
    main()
