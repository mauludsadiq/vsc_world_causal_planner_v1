from __future__ import annotations

import json
import sys
from pathlib import Path

from text_world.render_parse_clean_api import render_state_clean, parse_state_clean
from text_world.render_parse_enhanced import EnhancedRenderParse

def main():
    out_json = "results/smoke/text_enhanced_sentence_demo.json"
    n = 256
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        n = int(sys.argv[2])

    rp = EnhancedRenderParse(render_state_clean, parse_state_clean)

    ok = 0
    bad = []
    for sid in range(n):
        txt = rp.render(sid, strength=0.75, sugar=True)
        sid2, conf = rp.parse(txt)
        if sid2 == sid:
            ok += 1
        else:
            bad.append({"sid": sid, "txt": txt, "parsed_sid": sid2, "conf": conf})

    report = {
        "TEXT_ENHANCED_SENTENCE_DEMO": {
            "n": n,
            "ok": ok,
            "fail": len(bad),
            "fails": bad[:20],
        }
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if len(bad) != 0:
        raise SystemExit(f"[FAIL] TEXT_ENHANCED_SENTENCE_ROUNDTRIP fail={len(bad)} ok={ok} n={n}")

    print(f"[PASS] TEXT_ENHANCED_SENTENCE_ROUNDTRIP ok={ok} n={n}")

if __name__ == "__main__":
    main()
