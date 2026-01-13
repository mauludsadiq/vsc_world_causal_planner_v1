export TEXT_FAST=1
set -euo pipefail
export PYTHONPATH="$(pwd)"

PY="${PYTHON:-python}"
SEED="${1:-0}"
N_BLOCK="${2:-32}"

bash tools/run_all_text_demos.sh "$SEED" "$N_BLOCK"
