from __future__ import annotations
import sys
import json
from pathlib import Path

from text_world.agent.self_prompt_loop import run_self_prompt_loop
from text_world.agent.block_search import run_block_beam_search
from text_world.agent.personality_contract import emit_personality_reply

def main() -> None:
    out_json = "results/text_personality_agent.json"
    seed = 0
    epsilon = 0.15
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    if len(sys.argv) >= 4:
        epsilon = float(sys.argv[3])

    tmp_loop = "results/_tmp_loop.json"
    tmp_beam = "results/_tmp_beam.json"

    run_self_prompt_loop(tmp_loop, seed=seed, epsilon=epsilon, horizon=6)
    beam = run_block_beam_search(tmp_beam, seed=seed, epsilon=epsilon, depth=6, beam=6)

    loop = json.loads(Path(tmp_loop).read_text(encoding="utf-8"))
    best = beam["BLOCK_BEAM_SEARCH"]

    main_reply = f"Chose path {best['best_path']} because estimated value={best['best_value']:.4f} and risk_max={best['best_risk_max']:.4f}<=epsilon={epsilon}. Final text-state: {best['best_text']}"

    safety = {
        "epsilon": epsilon,
        "risk_max": best["best_risk_max"],
        "ok": (best["best_risk_max"] <= epsilon),
    }

    rejected = best["rejected_example"]
    if rejected is None:
        rejected = {
            "note": "No rejected branch found in beam frontier; expanding beam/depth will produce one.",
            "risk": None,
            "epsilon": epsilon,
        }
    else:
        rejected = {
            "if_chosen_path": rejected["path"],
            "expected_outcome_value": rejected["value"],
            "risk_max": rejected["risk_max"],
            "epsilon": epsilon,
            "verdict": "rejected" if rejected["risk_max"] > epsilon else "accepted",
        }

    emit_personality_reply(main_reply=main_reply, safety=safety, rejected=rejected, out_json=out_json)

if __name__ == "__main__":
    main()
