set -euo pipefail

PY="${PYTHON:-python}"
SEED="${1:-0}"
N_BLOCK="${2:-32}"

mkdir -p results

run() {
  echo
  echo "==> $*"
  "$@"
}

run $PY -m experiments.text_sentence_demo results/text_sentence_demo.json "$SEED"
run $PY -m experiments.text_paragraph_world_demo results/text_paragraph_world_demo.json "$SEED"
run $PY -m experiments.text_document_tradeoff_demo results/text_document_tradeoff_demo.json "$SEED"
run $PY -m experiments.text_block_world_demo results/text_block_world_demo.json "$SEED" "$N_BLOCK"
run $PY -m experiments.text_block_tradeoff_demo results/text_block_tradeoff_demo.json "$SEED" "$N_BLOCK"
run $PY -m experiments.text_block_counterfactual_explain_demo results/text_block_counterfactual_explain_demo.json "$SEED" "$N_BLOCK" 5 4
run $PY -m experiments.text_block_complex_sweep_demo results/text_block_complex_sweep_demo.json "$SEED"
run $PY -m experiments.text_full_stack_demo results/text_full_stack_demo.json "$SEED" "$N_BLOCK"

echo
echo "[PASS] RUN_ALL_TEXT_DEMOS_DONE"
echo "WROTE:"
ls -1 results/text_*_demo*.json | sed 's/^/  - /'

$PY -m experiments.text_self_prompt_loop_demo results/text_self_prompt_loop.json "$SEED" 0.15 "$N_BLOCK"
$PY -m experiments.text_grammar_bootstrap_demo results/text_grammar_bootstrap.json compose_v0 50
$PY -m experiments.text_block_beam_search_demo results/text_block_beam_search.json "$SEED" 0.15 "$N_BLOCK" 8
$PY -m experiments.text_personality_agent_demo results/text_personality_agent.json "$SEED" 0.15

python -m experiments.text_block_hier_demo results/text_block_hier_demo.json "$SEED" "$N_BLOCK" 6
