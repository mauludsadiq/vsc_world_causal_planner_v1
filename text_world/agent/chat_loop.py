from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from text_world.agent.dialogue_proof import DecodeProof, TurnProof, build_dialogue_proof


@dataclass(frozen=True)
class AgentConfig:
    seed: int
    epsilon: float = 0.12
    neural_model_dir: Optional[str] = None
    strength: str = "plain"
    symbolic_first: bool = True
    tau_p: float = 0.90
    tau_margin: float = 0.10


def _strength_to_float(s: str) -> float:
    x = str(s).strip().lower()
    if x in ("plain", "normal", "default"):
        return 0.75
    if x in ("firm", "strong"):
        return 0.95
    if x in ("soft", "gentle"):
        return 0.55
    return 0.75


def _run_dialogue_script_v2(
    inputs: Sequence[str],
    *,
    config: AgentConfig,
    parser_clean: Callable[[str], Optional[int]],
    render_sentence: Callable[..., str],
    decode_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = config
    seed_i = int(cfg.seed)
    strength_f = _strength_to_float(cfg.strength)

    neural = None
    if cfg.neural_model_dir is not None:
        try:
            from text_world.neural_inverse import load_neural_inverse
            neural = load_neural_inverse(str(cfg.neural_model_dir))
        except Exception:
            neural = None

    turns: List[TurnProof] = []
    sid = 0

    for t, user_text in enumerate(list(inputs)):
        user_text = str(user_text)

        d: Dict[str, Any]
        try:
            d = decode_fn(
                user_text,
                seed=seed_i,
                symbolic_first=bool(cfg.symbolic_first),
                parser_clean=parser_clean,
                neural=neural,
                tau_p=float(cfg.tau_p),
                tau_margin=float(cfg.tau_margin),
            )
        except TypeError:
            try:
                d = decode_fn(
                    user_text,
                    seed_i,
                    bool(cfg.symbolic_first),
                    parser_clean,
                    neural,
                    float(cfg.tau_p),
                    float(cfg.tau_margin),
                )
            except TypeError:
                d = decode_fn(user_text, seed_i)

        mode = str(d.get("mode"))
        sid_hat = d.get("sid_hat", None)

        dp = DecodeProof(
            mode=mode,
            sid_hat=(int(sid_hat) if sid_hat is not None else None),
            p_top1=(float(d["p_top1"]) if "p_top1" in d and d["p_top1"] is not None else None),
            p_top2=(float(d["p_top2"]) if "p_top2" in d and d["p_top2"] is not None else None),
            margin=(float(d["margin"]) if "margin" in d and d["margin"] is not None else None),
            entropy=(float(d["entropy"]) if "entropy" in d and d["entropy"] is not None else None),
            tau_p=float(d.get("tau_p", cfg.tau_p)),
            tau_margin=float(d.get("tau_margin", cfg.tau_margin)),
            seed=int(d.get("seed", seed_i)),
            reason=(str(d["reason"]) if "reason" in d else None),
        )

        assistant_text: Optional[str] = None
        sid_out: Optional[int] = None

        if mode in ("symbolic", "neural") and dp.sid_hat is not None:
            sid_out = int(dp.sid_hat)
            try:
                assistant_text = str(render_sentence(sid_out, strength=strength_f))
            except TypeError:
                assistant_text = str(render_sentence(sid_out))

        turns.append(
            TurnProof(
                turn=int(t),
                user_text=str(user_text),
                decode=dp,
                sid_in=int(sid),
                sid_out=(int(sid_out) if sid_out is not None else None),
                assistant_text=(str(assistant_text) if assistant_text is not None else None),
                main_reply=(str(assistant_text) if assistant_text is not None else None),
                safety_verdict={
                    "epsilon": float(cfg.epsilon),
                    "verdict": ("allow" if (sid_out is not None) else "reject"),
                    "risk": (0.0 if (sid_out is not None) else 1.0),
                },
                rejected_counterfactuals=([] if (sid_out is not None) else [
                    {
                        "reason": "decode_reject",
                        "sid_hat": (int(dp.sid_hat) if dp.sid_hat is not None else None),
                        "mode": str(dp.mode),
                    }
                ]),
            )
        )

        if sid_out is not None:
            sid = int(sid_out)

    art = build_dialogue_proof(seed_i, turns)

    art["agent_cfg"] = {
        "seed": int(cfg.seed),
        "epsilon": float(cfg.epsilon),
        "neural_model_dir": (str(cfg.neural_model_dir) if cfg.neural_model_dir is not None else None),
        "strength": str(cfg.strength),
        "symbolic_first": bool(cfg.symbolic_first),
        "tau_p": float(cfg.tau_p),
        "tau_margin": float(cfg.tau_margin),
    }

    return art


def _run_dialogue_script_legacy(
    *,
    seed: int,
    n_turns: int,
    render_fn: Callable[[int], str],
    decode_fn: Callable[..., Dict[str, Any]],
    inputs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    seed_i = int(seed)
    n = int(n_turns)

    if inputs is None:
        inputs = ["the claim is unspecified", "unknown", "the claim is unspecified"][:n]
        if len(inputs) < n:
            inputs = list(inputs) + ["unknown"] * (n - len(inputs))

    turns: List[TurnProof] = []
    sid = 0

    for t in range(n):
        user_text = str(list(inputs)[t])

        d = decode_fn(user_text, seed_i)

        mode = str(d.get("mode"))
        sid_hat = d.get("sid_hat", None)

        dp = DecodeProof(
            mode=mode,
            sid_hat=(int(sid_hat) if sid_hat is not None else None),
            p_top1=(float(d["p_top1"]) if "p_top1" in d and d["p_top1"] is not None else None),
            p_top2=(float(d["p_top2"]) if "p_top2" in d and d["p_top2"] is not None else None),
            margin=(float(d["margin"]) if "margin" in d and d["margin"] is not None else None),
            entropy=(float(d["entropy"]) if "entropy" in d and d["entropy"] is not None else None),
            tau_p=float(d.get("tau_p", 0.90)),
            tau_margin=float(d.get("tau_margin", 0.10)),
            seed=int(d.get("seed", seed_i)),
            reason=(str(d["reason"]) if "reason" in d else None),
        )

        sid_out: Optional[int] = None
        assistant_text: Optional[str] = None

        if mode in ("symbolic", "neural") and dp.sid_hat is not None:
            sid_out = int(dp.sid_hat)
            assistant_text = str(render_fn(sid_out))

        turns.append(
            TurnProof(
                turn=int(t),
                user_text=str(user_text),
                decode=dp,
                sid_in=int(sid),
                sid_out=(int(sid_out) if sid_out is not None else None),
                assistant_text=(str(assistant_text) if assistant_text is not None else None),
                main_reply=(str(assistant_text) if assistant_text is not None else None),
                safety_verdict={
                    "epsilon": 0.12,
                    "verdict": ("allow" if (sid_out is not None) else "reject"),
                    "risk": (0.0 if (sid_out is not None) else 1.0),
                },
                rejected_counterfactuals=([] if (sid_out is not None) else [
                    {
                        "reason": "decode_reject",
                        "sid_hat": (int(dp.sid_hat) if dp.sid_hat is not None else None),
                        "mode": str(dp.mode),
                    }
                ]),
            )
        )

        if sid_out is not None:
            sid = int(sid_out)

    out = build_dialogue_proof(seed_i, turns)
    out["n_turns"] = int(n)
    return out


def run_dialogue_script(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    if len(args) >= 1 and isinstance(args[0], (list, tuple)):
        return _run_dialogue_script_v2(*args, **kwargs)

    if "config" in kwargs or "parser_clean" in kwargs or "render_sentence" in kwargs:
        return _run_dialogue_script_v2(*args, **kwargs)

    if "seed" in kwargs and "n_turns" in kwargs and "render_fn" in kwargs and "decode_fn" in kwargs:
        return _run_dialogue_script_legacy(**kwargs)

    raise TypeError("run_dialogue_script: unsupported call signature")
