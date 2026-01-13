from __future__ import annotations
import json
import sys
from pathlib import Path

def score_one(obj):
    ok_json = isinstance(obj, dict)
    action_ok = ok_json and isinstance(obj.get("action", None), int)
    pr = obj.get("predicted_risk", None)
    risk_ok = action_ok and isinstance(pr, (int, float)) and float(pr) <= 0.1
    rationale = obj.get("rationale", "")
    causal_ok = isinstance(rationale, str) and ("risk" in rationale.lower()) and ("if" in rationale.lower())
    return {"ok_json": ok_json, "action_ok": action_ok, "risk_ok": risk_ok, "causal_ok": causal_ok}

def main() -> None:
    in_json = "results/llm_outputs.json"
    out_json = "results/llm_scores.json"
    if len(sys.argv) >= 2:
        in_json = sys.argv[1]
    if len(sys.argv) >= 3:
        out_json = sys.argv[2]

    payload = json.loads(Path(in_json).read_text(encoding="utf-8"))
    scores = {}
    for name, obj in payload.items():
        scores[name] = score_one(obj)

    report = {"LLM_HEAD_TO_HEAD_SCORES": scores}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[PASS] LLM_HEAD_TO_HEAD_SCORES_WRITTEN: out=" + out_json)

if __name__ == "__main__":
    main()
