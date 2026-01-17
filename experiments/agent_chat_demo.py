from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from text_world.agent.chat_loop import run_dialogue_script
from text_world.render_parse_enhanced import render_sentence_enhanced

from text_world.neural_inverse import load_neural_inverse, decode_text_to_sid


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _stable_json(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def main() -> int:
    import sys

    if len(sys.argv) not in (3, 4):
        print("usage: python -m experiments.agent_chat_demo OUT_JSON SEED [NEURAL_MODEL_DIR]")
        return 2

    out_path = Path(sys.argv[1])
    seed = int(sys.argv[2])
    neural_dir: Optional[str] = sys.argv[3] if len(sys.argv) == 4 else None

    neural = load_neural_inverse(neural_dir) if neural_dir else None

    def decode_fn(text: str, seed: int) -> Dict[str, Any]:
        from text_world.render_parse_clean import parse_sentence_clean

        return decode_text_to_sid(
            str(text),
            seed=int(seed),
            symbolic_first=True,
            parser_clean=parse_sentence_clean,
            neural=neural,
            tau_p=0.90,
            tau_margin=0.10,
        )

    trace = run_dialogue_script(
        seed=seed,
        n_turns=4,
        render_fn=lambda sid: render_sentence_enhanced(int(sid)),
        decode_fn=decode_fn,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(_stable_json(trace))

    sha = _sha256_bytes(out_path.read_bytes())
    print(f"[PASS] AGENT_CHAT_DEMO: n_turns={trace['n_turns']} sha256={sha} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
