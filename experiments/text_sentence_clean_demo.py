from __future__ import annotations
import json
import sys
from pathlib import Path

from text_world.state import enumerate_states
from text_world.render_parse_clean import render_sentence_sidecar, parse_sentence_clean

def run(out_json: str) -> dict:
    states = enumerate_states()
    failures = []
    examples = []

    for st in states:
        text, tag = render_sentence_sidecar(st)
        st2 = parse_sentence_clean(text)
        if st2 != st:
            failures.append({
                "expected": st.__dict__,
                "got": st2.__dict__,
                "text": text,
                "tag": tag,
            })
        if len(examples) < 8:
            examples.append({"text": text, "tag": tag})

    report = {
        "n_states": len(states),
        "n_failures": len(failures),
        "examples": examples,
        "failures": failures[:5],
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failures:
        raise SystemExit(1)

    print(f"[PASS] TEXT_SENTENCE_CLEAN_RENDER_PARSE_ROUNDTRIP: n_states={len(states)}")
    return report

def main() -> None:
    out_json = "results/text_sentence_clean_demo.json"
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    run(out_json)

if __name__ == "__main__":
    main()
