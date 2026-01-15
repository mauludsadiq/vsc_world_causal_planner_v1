from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from text_world.render_parse_clean_api import render_state_clean, parse_state_clean
from text_world.render_parse_enhanced import EnhancedRenderParse


def main() -> int:
    if len(sys.argv) != 4:
        print("USAGE: python -m experiments.gen_enhanced_sentence_pairs OUT.jsonl N SEED", file=sys.stderr)
        return 2

    out_path = Path(sys.argv[1])
    n = int(sys.argv[2])
    seed = int(sys.argv[3])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rp = EnhancedRenderParse(render_state_clean, parse_state_clean)

    rng = random.Random(seed)
    strengths = [0.15, 0.35, 0.50, 0.65, 0.80, 0.95]

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            sid = rng.randrange(256)
            strength = rng.choice(strengths)
            txt = rp.render(sid, strength=strength, sugar=True)

            rec = {
                "sid": sid,
                "text": txt,
                "conf": float(strength),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[PASS] GEN_ENHANCED_SENTENCE_PAIRS: wrote={out_path} n={n} seed={seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
