from __future__ import annotations

from typing import Any, Dict, List, Optional

from text_world.agent.chat_loop import AgentConfig, run_dialogue_script


def _fake_parser_clean(text: str) -> Optional[int]:
    if text.strip() == "the claim is unspecified":
        return 7
    return None


def _fake_render_sentence(sid: int) -> str:
    return f"SID={sid}"


def _fake_decode_fn(
    text: str,
    *,
    seed: int,
    symbolic_first: bool,
    parser_clean,
    neural,
    tau_p: float,
    tau_margin: float,
) -> Dict[str, Any]:
    sid = parser_clean(text) if symbolic_first else None
    if sid is not None:
        return {
            "mode": "symbolic",
            "sid_hat": int(sid),
            "p_top1": 1.0,
            "p_top2": 0.0,
            "margin": 1.0,
            "entropy": 0.0,
            "tau_p": float(tau_p),
            "tau_margin": float(tau_margin),
            "seed": int(seed),
        }
    return {
        "mode": "reject",
        "sid_hat": None,
        "reason": "no_parse_no_neural",
        "tau_p": float(tau_p),
        "tau_margin": float(tau_margin),
        "seed": int(seed),
    }


def test_agent_trace_is_deterministic() -> None:
    inputs = ["the claim is unspecified", "unknown"]
    cfg = AgentConfig(seed=0, epsilon=0.12, neural_model_dir=None, strength="firm")

    t1 = run_dialogue_script(
        inputs,
        config=cfg,
        parser_clean=_fake_parser_clean,
        render_sentence=_fake_render_sentence,
        decode_fn=_fake_decode_fn,
    )

    t2 = run_dialogue_script(
        inputs,
        config=cfg,
        parser_clean=_fake_parser_clean,
        render_sentence=_fake_render_sentence,
        decode_fn=_fake_decode_fn,
    )

    assert t1 == t2
    assert t1["seed"] == 0
    assert len(t1["turns"]) == 2


def test_agent_turn_schema() -> None:
    inputs = ["the claim is unspecified", "unknown"]
    cfg = AgentConfig(seed=0, epsilon=0.12, neural_model_dir=None, strength="plain")

    tr = run_dialogue_script(
        inputs,
        config=cfg,
        parser_clean=_fake_parser_clean,
        render_sentence=_fake_render_sentence,
        decode_fn=_fake_decode_fn,
    )

    for turn in tr["turns"]:
        assert "turn" in turn
        assert "user_text" in turn
        assert "decode" in turn
        assert "sid_out" in turn
        assert "main_reply" in turn
        assert "safety_verdict" in turn
        assert "rejected_counterfactuals" in turn
        assert "mode" in turn["decode"]
