from __future__ import annotations

import json
import sys
from pathlib import Path

from text_world.render_parse_clean_api import render_state_clean, parse_state_clean
from text_world.render_parse_enhanced import EnhancedRenderParse


def main() -> int:
    if len(sys.argv) != 2:
        print("USAGE: python -m experiments.verify_enhanced_dataset_roundtrip DATA.jsonl", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    rp = EnhancedRenderParse(render_state_clean, parse_state_clean)

    n = 0
    n_fail = 0
    max_conf_err = 0.0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = int(rec["sid"])
            txt = str(rec["text"])
            conf0 = float(rec.get("conf", 1.0))

            sid2, conf2 = rp.parse(txt)

            ok = (sid2 == sid)
            if not ok:
                n_fail += 1

            conf_err = abs(conf2 - conf0)
            if conf_err > max_conf_err:
                max_conf_err = conf_err

            n += 1

    if n_fail == 0:
        print(f"[PASS] VERIFY_ENHANCED_DATASET_ROUNDTRIP: n={n} max_conf_err={max_conf_err:.6f} path={path}")
        return 0

    print(f"[FAIL] VERIFY_ENHANCED_DATASET_ROUNDTRIP: n={n} failures={n_fail} max_conf_err={max_conf_err:.6f} path={path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
