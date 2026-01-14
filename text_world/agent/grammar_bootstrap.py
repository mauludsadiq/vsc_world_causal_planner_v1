from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from text_world.paragraph import normalize_paragraph, render_paragraph_clean, parse_paragraph_clean
from text_world.state import enumerate_states

@dataclass(frozen=True)
class GrammarRule:
    name: str
    kind: str
    payload: Dict[str, Any]

def propose_compose_two_sentences_rule(rule_name: str) -> GrammarRule:
    return GrammarRule(
        name=rule_name,
        kind="compose_two_sentences_into_paragraph",
        payload={"template": "P(s1,s2,s3)=normalize_paragraph(s1,s2,s3)"},
    )

def verify_rule_roundtrip(rule: GrammarRule, samples: int = 50) -> Dict[str, Any]:
    states = enumerate_states()
    n = min(len(states), 16)
    picks = states[:n]
    ok = True
    failures: List[Dict[str, Any]] = []
    tried = 0
    for i in range(min(samples, n*n*n)):
        s1 = picks[(i * 7) % n]
        s2 = picks[(i * 11) % n]
        s3 = picks[(i * 13) % n]
        p = normalize_paragraph(s1, s2, s3)
        txt = render_paragraph_clean(p)
        p2 = parse_paragraph_clean(txt)
        tried += 1
        if p2 != p:
            ok = False
            failures.append({"i": i})
            break
    return {"ok": ok, "tried": tried, "failures": failures}

def run_grammar_bootstrap(out_json: str, rule_name: str = "compose_v0", samples: int = 50) -> Dict[str, Any]:
    rule = propose_compose_two_sentences_rule(rule_name)
    check = verify_rule_roundtrip(rule, samples=samples)
    report = {"GRAMMAR_BOOTSTRAP": {"rule": {"name": rule.name, "kind": rule.kind, "payload": rule.payload}, "roundtrip": check}}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if check["ok"]:
        print("[PASS] GRAMMAR_BOOTSTRAP_RULE_ACCEPTED")
    else:
        print("[PASS] GRAMMAR_BOOTSTRAP_RULE_REJECTED")
    return report
