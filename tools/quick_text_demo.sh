set -euo pipefail
PY="${PYTHON:-python}"
SEED="${1:-0}"
N_BLOCK="${2:-8}"
export TEXT_DEMO_FAST=1
PYTHONUNBUFFERED=1 "$PY" -u -X faulthandler -m experiments.text_full_stack_demo results/text_full_stack_demo.json "$SEED" "$N_BLOCK"
cat results/text_full_stack_demo.json
