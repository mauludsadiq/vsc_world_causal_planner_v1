from __future__ import annotations
import sys
from text_world.agent.grammar_bootstrap import run_grammar_bootstrap

def main() -> None:
    out_json = "results/text_grammar_bootstrap.json"
    rule_name = "compose_v0"
    samples = 50
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        rule_name = sys.argv[2]
    if len(sys.argv) >= 4:
        samples = int(sys.argv[3])
    run_grammar_bootstrap(out_json, rule_name=rule_name, samples=samples)

if __name__ == "__main__":
    main()
