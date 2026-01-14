from __future__ import annotations
import sys
from text_world.agent.self_prompt_loop import run_self_prompt_loop

def main() -> None:
    out_json = "results/text_self_prompt_loop.json"
    seed = 0
    epsilon = 0.15
    horizon = 8
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    if len(sys.argv) >= 4:
        epsilon = float(sys.argv[3])
    if len(sys.argv) >= 5:
        horizon = int(sys.argv[4])
    run_self_prompt_loop(out_json, seed=seed, epsilon=epsilon, horizon=horizon)

if __name__ == "__main__":
    main()
