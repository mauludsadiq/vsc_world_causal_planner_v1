set -euo pipefail

PY="${PYTHON:-python}"
SEED="${1:-0}"
N_BLOCK="${2:-32}"

$PY -m pytest -q tests_text
bash tools/run_all_text_demos.sh "$SEED" "$N_BLOCK"
