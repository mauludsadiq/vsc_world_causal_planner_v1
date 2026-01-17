from __future__ import annotations

from typing import Any, Dict

from text_world.agent.chat_loop import run_dialogue_script


def test_agent_dialogue_proof_shape_minimal() -> None:
    def render_fn(sid: int) -> str:
        return f"state={sid}"

    def decode_fn(text: str, seed: int) -> Dict[str, Any]:
        return {
            "mode": "symbolic",
            "sid_hat": 0,
            "p_top1": 1.0,
            "p_top2": 0.0,
            "margin": 1.0,
            "entropy": 0.0,
            "tau_p": 0.90,
            "tau_margin": 0.10,
            "seed": int(seed),
        }

    out = run_dialogue_script(seed=0, n_turns=3, render_fn=render_fn, decode_fn=decode_fn)

    assert isinstance(out, dict)
    assert out["seed"] == 0
    assert out["n_turns"] == 3
    assert isinstance(out["turns"], list)
    assert len(out["turns"]) == 3
    assert "sha256" in out

    t0 = out["turns"][0]
    assert "decoded" in t0
    assert t0["decoded"]["mode"] in ("symbolic", "neural", "reject")
