set -euo pipefail

PY="${PYTHON:-python}"
SEED="${1:-0}"
N_BLOCK="${2:-32}"

mkdir -p results

$PY -m experiments.text_sentence_demo results/text_sentence_demo.json "$SEED"
$PY -m experiments.text_paragraph_world_demo results/text_paragraph_world_demo.json "$SEED"
$PY -m experiments.text_document_tradeoff_demo results/text_document_tradeoff_demo.json "$SEED"
$PY -m experiments.text_block_world_demo results/text_block_world_demo.json "$SEED" "$N_BLOCK"
$PY -m experiments.text_block_tradeoff_demo results/text_block_tradeoff_demo.json "$SEED" "$N_BLOCK"
$PY -m experiments.text_block_counterfactual_explain_demo results/text_block_counterfactual_explain_demo.json "$SEED" "$N_BLOCK" 5 4
$PY -m experiments.text_block_complex_sweep_demo results/text_block_complex_sweep_demo.json "$SEED"
$PY -m experiments.text_full_stack_demo results/text_full_stack_demo.json "$SEED" "$N_BLOCK"

echo
echo "[PASS] RUN_ALL_TEXT_DEMOS_DONE"
echo "WROTE:"
ls -1 results/text_*_demo*.json | sed 's/^/  - /'
