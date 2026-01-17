from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from text_world.agent.dialogue_proof import DecodeProof, TurnProof, build_dialogue_proof


def run_dialogue_script(
    *,
    seed: int,
    n_turns: int,
    render_fn: Callable[[int], str],
    decode_fn: Callable[[str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Runs a deterministic dialogue script.

    Proof-carrying artifact:
      - each turn records the raw user_text
      - decode proof (symbolic vs neural vs reject)
      - state_id_in/out
      - assistant_text (rendered from chosen state_id_out if any)
    """
    turns = []

    state_id = 0

    for t in range(int(n_turns)):
        user_text = f"user turn {t}"
        d = decode_fn(user_text, int(seed))

        mode = str(d.get("mode"))
        sid_hat = d.get("sid_hat", None)

        dp = DecodeProof(
            mode=mode,
            sid_hat=(int(sid_hat) if sid_hat is not None else None),
            p_top1=(float(d["p_top1"]) if "p_top1" in d and d["p_top1"] is not None else None),
            p_top2=(float(d["p_top2"]) if "p_top2" in d and d["p_top2"] is not None else None),
            margin=(float(d["margin"]) if "margin" in d and d["margin"] is not None else None),
            entropy=(float(d["entropy"]) if "entropy" in d and d["entropy"] is not None else None),
            tau_p=float(d.get("tau_p", 0.0)),
            tau_margin=float(d.get("tau_margin", 0.0)),
            seed=int(d.get("seed", seed)),
            reason=(str(d["reason"]) if "reason" in d else None),
        )

        if mode in ("symbolic", "neural") and dp.sid_hat is not None:
            state_id_out: Optional[int] = int(dp.sid_hat)
            assistant_text = str(render_fn(state_id_out))
        else:
            state_id_out = None
            assistant_text = None

        turns.append(
            TurnProof(
                turn=int(t),
                user_text=str(user_text),
                decoded=dp,
                state_id_in=int(state_id),
                state_id_out=(int(state_id_out) if state_id_out is not None else None),
                assistant_text=(str(assistant_text) if assistant_text is not None else None),
            )
        )

        if state_id_out is not None:
            state_id = int(state_id_out)

    return build_dialogue_proof(int(seed), turns)
