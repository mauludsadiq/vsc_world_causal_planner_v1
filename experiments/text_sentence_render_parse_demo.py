from __future__ import annotations
import json
import sys
from pathlib import Path

from text_world.state import enumerate_states
from text_world.render_parse import render_sentence, parse_sentence

def run(out_json: str) -> dict:
    states = enumerate_states()
    failures = []
    examples = []

    for st in states:
        sent = render_sentence(st)
        st2 = parse_sentence(sent)
        if st2 != st:
            failures.append({
                "expected": st.__dict__,
                "got": st2.__dict__,
                "sentence": sent,
            })
        if len(examples) < 8:
            examples.append(sent)

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

    print(f"[PASS] TEXT_SENTENCE_RENDER_PARSE_ROUNDTRIP: n_states={len(states)}")
    return report

def main() -> None:
    out_json = "results/text_sentence_render_parse_demo.json"
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    run(out_json)

if __name__ == "__main__":
    main()
