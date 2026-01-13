from __future__ import annotations
import json
import sys
from pathlib import Path
try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    _MPL_ERR = str(e)


def main() -> None:
    in_json = "results/text_full_stack_demo.json"
    out_dir = "artifacts/full_stack_viz"
    if len(sys.argv) >= 2:
        in_json = sys.argv[1]
    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]

    data = json.loads(Path(in_json).read_text(encoding="utf-8"))
    h = data["highlights"]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if plt is None:
        out = {
            "TEXT_FULL_STACK_VIZ": {
                "in_json": in_json,
                "out_dir": out_dir,
                "files": [],
                "skipped": True,
                "reason": "matplotlib not installed: " + _MPL_ERR,
            }
        }
        viz_json = str(Path(out_dir) / "viz.json")
        Path(viz_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("[PASS] TEXT_FULL_STACK_VIZ_WRITTEN: out=" + viz_json)
        return


    l1_items = []
    if "TEXT_WORLD_MODEL_TRANSITION_L1" in h:
        l1_items.append(("sentence", h["TEXT_WORLD_MODEL_TRANSITION_L1"]["mean_l1"]))
    if "PARA_WORLD_MODEL_TRANSITION_L1" in h:
        l1_items.append(("paragraph", h["PARA_WORLD_MODEL_TRANSITION_L1"]["mean_l1"]))
    if "BLOCK_WORLD_MODEL_TRANSITION_L1" in h:
        l1_items.append(("block", h["BLOCK_WORLD_MODEL_TRANSITION_L1"]["mean_l1"]))

    if l1_items:
        plt.figure()
        plt.bar([x[0] for x in l1_items], [x[1] for x in l1_items])
        plt.title("Mean L1 by scale")
        p = str(Path(out_dir) / "mean_l1.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()

    risk_items = []
    if "TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED" in h:
        risk_items.append(("sentence_opt_risk", h["TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED"]["opt_risk"]))
        risk_items.append(("sentence_eps", h["TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED"]["epsilon"]))
    if "DOC_SAFETY_TRADEOFF_FORCED" in h:
        risk_items.append(("doc_opt_risk", h["DOC_SAFETY_TRADEOFF_FORCED"]["opt_risk"]))
        risk_items.append(("doc_eps", h["DOC_SAFETY_TRADEOFF_FORCED"]["epsilon"]))
    if "BLOCK_SAFETY_TRADEOFF_FORCED" in h:
        risk_items.append(("block_opt_risk", h["BLOCK_SAFETY_TRADEOFF_FORCED"]["opt_risk"]))
        risk_items.append(("block_eps", h["BLOCK_SAFETY_TRADEOFF_FORCED"]["epsilon"]))

    if risk_items:
        plt.figure()
        plt.bar([x[0] for x in risk_items], [x[1] for x in risk_items])
        plt.title("Risk vs epsilon")
        p = str(Path(out_dir) / "risk_vs_eps.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()

    ret_items = []
    if "DOC_SAFETY_TRADEOFF_FORCED" in h:
        ret_items.append(("doc_opt_return", h["DOC_SAFETY_TRADEOFF_FORCED"]["opt_return"]))
        ret_items.append(("doc_chosen_return", h["DOC_SAFETY_TRADEOFF_FORCED"]["chosen_return"]))
    if "BLOCK_SAFETY_TRADEOFF_FORCED" in h:
        ret_items.append(("block_opt_return", h["BLOCK_SAFETY_TRADEOFF_FORCED"]["opt_return"]))
        ret_items.append(("block_chosen_return", h["BLOCK_SAFETY_TRADEOFF_FORCED"]["chosen_return"]))

    if ret_items:
        plt.figure()
        plt.bar([x[0] for x in ret_items], [x[1] for x in ret_items])
        plt.title("Return: opt vs chosen")
        p = str(Path(out_dir) / "return_opt_vs_chosen.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()

    out = {
        "TEXT_FULL_STACK_VIZ": {
            "in_json": in_json,
            "out_dir": out_dir,
            "files": [str(p) for p in Path(out_dir).glob("*.png")],
        }
    }
    viz_json = str(Path(out_dir) / "viz.json")
    Path(viz_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[PASS] TEXT_FULL_STACK_VIZ_WRITTEN: out=" + viz_json)

if __name__ == "__main__":
    main()
