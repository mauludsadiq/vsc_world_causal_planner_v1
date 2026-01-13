from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(p.stderr or f"command failed: {cmd}")
    return p.stdout

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main() -> None:
    # usage:
    #   python -m experiments.text_full_stack_demo results/text_full_stack_demo.json 0
    out_json = "results/text_full_stack_demo.json"
    seed = "0"
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = sys.argv[2]

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    sent_json = "results/text_sentence_demo.json"
    para_json = "results/text_paragraph_world_demo.json"
    doc_json  = "results/text_document_tradeoff_demo.json"

    # Run the three demos. We intentionally forward their stdout PASS lines unchanged.
    sent_out = run_cmd([sys.executable, "-m", "experiments.text_sentence_demo", sent_json, seed])
    sys.stdout.write(sent_out)

    para_out = run_cmd([sys.executable, "-m", "experiments.text_paragraph_world_demo", para_json, seed])
    sys.stdout.write(para_out)

    doc_out = run_cmd([sys.executable, "-m", "experiments.text_document_tradeoff_demo", doc_json, seed])
    sys.stdout.write(doc_out)

    # Build a compact summary (plus pointers to the full per-demo JSON).
    sent = load_json(sent_json)
    para = load_json(para_json)
    doc  = load_json(doc_json)

    summary = {
        "seed": int(seed),
        "artifacts": {
            "sentence_json": sent_json,
            "paragraph_json": para_json,
            "document_json": doc_json,
        },
        "highlights": {
            "TEXT_SCM_BACKDOOR": {
                "max_abs_err_backdoor": sent["TEXT_SCM_BACKDOOR"]["max_abs_err_backdoor"],
                "tol": sent["TEXT_SCM_BACKDOOR"]["tol"],
                "min_gap_naive": sent["TEXT_SCM_BACKDOOR"]["min_gap_naive"],
            },
            "TEXT_WORLD_MODEL_TRANSITION_L1": sent["TEXT_WORLD_MODEL_TRANSITION_L1"],
            "TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE": {
                "abs_return_diff": sent["TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE"]["abs_return_diff"],
                "vi_return": sent["TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE"]["vi_return"],
                "bf_return": sent["TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE"]["bf_return"],
            },
            "TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED": {
                "chosen_action": sent["TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED"]["chosen_action"],
                "chosen_risk": sent["TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED"]["chosen_risk"],
                "epsilon": sent["TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED"]["epsilon"],
                "opt_risk": sent["TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED"]["opt_risk"],
            },
            "PARA_WORLD_MODEL_TRANSITION_L1": para["PARA_WORLD_MODEL_TRANSITION_L1"],
            "DOC_SAFETY_TRADEOFF_FORCED": doc["DOC_SAFETY_TRADEOFF_FORCED"],
        },
    }

    Path(out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PASS] TEXT_FULL_STACK_SUMMARY_WRITTEN: out={out_json}")

if __name__ == "__main__":
    main()
