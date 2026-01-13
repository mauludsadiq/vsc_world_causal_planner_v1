from __future__ import annotations
import json
import sys
from pathlib import Path

from text_world.real_text_adapter import to_sentence_state
from text_world.render_parse_clean import render_sentence_clean
from text_world.env_sentence import build_sentence_world

def main() -> None:
    out_json = "results/text_real_input_full_stack_probe.json"
    in_path = "corpora/real_text_samples.txt"
    seed = 0
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        in_path = sys.argv[2]
    if len(sys.argv) >= 4:
        seed = int(sys.argv[3])

    raw = Path(in_path).read_text(encoding="utf-8")
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    mapped = []
    for line in lines:
        st = to_sentence_state(line)
        mapped.append({
            "raw": line,
            "state": {"M": st.fact_mask, "C": st.contradiction, "S": st.style, "L": st.length},
            "render": render_sentence_clean(st),
        })

    world = build_sentence_world()
    report = {"REAL_TEXT_PROBE": {"seed": seed, "n_lines": len(lines), "mapped": mapped, "sentence_world_n_states": len(world.states)}}

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[PASS] REAL_TEXT_PROBE_WRITTEN: out=" + out_json)

if __name__ == "__main__":
    main()
