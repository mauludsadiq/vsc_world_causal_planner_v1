from __future__ import annotations
import json
import random
import sys
from pathlib import Path

from text_world.state import enumerate_states
from text_world.paragraph import normalize_paragraph, render_paragraph_clean, parse_paragraph_clean

def run(out_json: str, seed: int) -> dict:
    rng = random.Random(seed)
    states = enumerate_states()

    failures = []
    examples = []

    # sample 200 random paragraphs (finite state space is huge; we certify by property-testing)
    for _ in range(200):
        s1 = rng.choice(states)
        s2 = rng.choice(states)
        s3 = rng.choice(states)
        p = normalize_paragraph(s1, s2, s3)
        txt = render_paragraph_clean(p)
        p2 = parse_paragraph_clean(txt)
        if p2 != p:
            failures.append({
                "expected": {
                    "s1": p.s1.__dict__,
                    "s2": p.s2.__dict__,
                    "s3": p.s3.__dict__,
                    "rho": p.rho,
                },
                "got": {
                    "s1": p2.s1.__dict__,
                    "s2": p2.s2.__dict__,
                    "s3": p2.s3.__dict__,
                    "rho": p2.rho,
                },
                "text": txt,
            })
        if len(examples) < 5:
            examples.append({"rho": p.rho, "text": txt})

    report = {
        "seed": seed,
        "samples": 200,
        "n_failures": len(failures),
        "examples": examples,
        "failures": failures[:3],
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failures:
        raise SystemExit(1)

    print(f"[PASS] TEXT_PARAGRAPH_RENDER_PARSE_ROUNDTRIP: samples=200 seed={seed}")
    return report

def main() -> None:
    out_json = "results/text_paragraph_demo.json"
    seed = 0
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    run(out_json, seed)

if __name__ == "__main__":
    main()
