from __future__ import annotations
import json
import sys
from pathlib import Path

from text_world.explain_counterfactual import explain_counterfactual_block

def main() -> None:
    out_json = "results/text_block_counterfactual_explain_demo.json"
    seed = 0
    n = 32
    chosen_action = 5
    alt_action = 4
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    if len(sys.argv) >= 4:
        n = int(sys.argv[3])
    if len(sys.argv) >= 5:
        chosen_action = int(sys.argv[4])
    if len(sys.argv) >= 6:
        alt_action = int(sys.argv[5])

    data = explain_counterfactual_block(seed=seed, n=n, chosen_action=chosen_action, alt_action=alt_action)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps({"BLOCK_COUNTERFACTUAL_EXPLANATION": data}, indent=2), encoding="utf-8")
    print("[PASS] BLOCK_COUNTERFACTUAL_EXPLANATION_WRITTEN: out=" + out_json)

if __name__ == "__main__":
    main()
