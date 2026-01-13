from __future__ import annotations
import json
import sys
from pathlib import Path

from experiments.text_sentence_demo import main as run_sentence
from experiments.text_paragraph_world_demo import main as run_paragraph_world
from experiments.text_document_tradeoff_demo import main as run_document_tradeoff
from experiments.text_block_world_demo import main as run_block_world
from experiments.text_block_tradeoff_demo import main as run_block_tradeoff
from experiments.text_block_counterfactual_explain_demo import main as run_cf

def _run(mod_main, argv):
    old = sys.argv[:]
    try:
        sys.argv = argv
        mod_main()
    finally:
        sys.argv = old

def main() -> None:
    out_json = "results/text_full_stack_demo.json"
    seed = 0
    n_block = 32
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    if len(sys.argv) >= 4:
        n_block = int(sys.argv[3])

    Path("results").mkdir(parents=True, exist_ok=True)

    sent_json = "results/text_sentence_demo.json"
    para_json = "results/text_paragraph_world_demo.json"
    doc_json = "results/text_document_tradeoff_demo.json"
    bw_json = "results/text_block_world_demo.json"
    bt_json = "results/text_block_tradeoff_demo.json"
    cf_json = "results/text_block_counterfactual_explain_demo.json"

    _run(run_sentence, ["-m", sent_json, str(seed)])
    _run(run_paragraph_world, ["-m", para_json, str(seed)])
    _run(run_document_tradeoff, ["-m", doc_json, str(seed)])
    _run(run_block_world, ["-m", bw_json, str(seed), str(n_block)])
    _run(run_block_tradeoff, ["-m", bt_json, str(seed), str(n_block)])
    _run(run_cf, ["-m", cf_json, str(seed), str(n_block), "5", "4"])

    s = json.loads(Path(sent_json).read_text(encoding="utf-8"))
    p = json.loads(Path(para_json).read_text(encoding="utf-8"))
    d = json.loads(Path(doc_json).read_text(encoding="utf-8"))
    bw = json.loads(Path(bw_json).read_text(encoding="utf-8"))
    bt = json.loads(Path(bt_json).read_text(encoding="utf-8"))
    cf = json.loads(Path(cf_json).read_text(encoding="utf-8"))

    highlights = {}
    for k in ("TEXT_SCM_BACKDOOR","TEXT_WORLD_MODEL_TRANSITION_L1","TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE","TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED"):
        highlights[k] = {kk: s[k][kk] for kk in s[k].keys() if kk in ("max_abs_err_backdoor","tol","min_gap_naive","mean_l1","samples","threshold","abs_return_diff","vi_return","bf_return","chosen_action","chosen_risk","epsilon","opt_risk")}
        if not highlights[k]:
            highlights[k] = s[k]

    highlights["PARA_WORLD_MODEL_TRANSITION_L1"] = p["PARA_WORLD_MODEL_TRANSITION_L1"]
    if "PARA_SAFETY_CONSTRAINT_POLICY_SELECTED" in p:
        highlights["PARA_SAFETY_CONSTRAINT_POLICY_SELECTED"] = p["PARA_SAFETY_CONSTRAINT_POLICY_SELECTED"]

    highlights["DOC_SAFETY_TRADEOFF_FORCED"] = d["DOC_SAFETY_TRADEOFF_FORCED"]
    highlights["BLOCK_WORLD_MODEL_TRANSITION_L1"] = bw["BLOCK_WORLD_MODEL_TRANSITION_L1"]
    highlights["BLOCK_SAFETY_TRADEOFF_FORCED"] = bt["BLOCK_SAFETY_TRADEOFF_FORCED"]
    highlights["BLOCK_COUNTERFACTUAL_EXPLANATION"] = cf["BLOCK_COUNTERFACTUAL_EXPLANATION"]

    data = {
        "seed": seed,
        "n_block": n_block,
        "artifacts": {
            "sentence_json": sent_json,
            "paragraph_json": para_json,
            "document_json": doc_json,
            "block_world_json": bw_json,
            "block_tradeoff_json": bt_json,
            "counterfactual_json": cf_json,
        },
        "highlights": highlights,
    }

    Path(out_json).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("[PASS] TEXT_FULL_STACK_SUMMARY_WRITTEN: out=" + out_json)
    print("[PASS] BLOCK_COUNTERFACTUAL_EXPLANATION_INCLUDED: out=" + cf_json)

if __name__ == "__main__":
    main()
