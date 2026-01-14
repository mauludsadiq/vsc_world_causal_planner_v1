from __future__ import annotations
import sys
from text_world.agent.block_search import run_block_beam_search

def main() -> None:
    out_json = "results/text_block_beam_search.json"
    seed = 0
    epsilon = 0.15
    depth = 10
    beam = 8
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    if len(sys.argv) >= 4:
        epsilon = float(sys.argv[3])
    if len(sys.argv) >= 5:
        depth = int(sys.argv[4])
    if len(sys.argv) >= 6:
        beam = int(sys.argv[5])
    run_block_beam_search(out_json, seed=seed, epsilon=epsilon, depth=depth, beam=beam)

if __name__ == "__main__":
    main()
