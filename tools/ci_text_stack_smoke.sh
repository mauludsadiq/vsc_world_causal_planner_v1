set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
SEED="${1:-0}"
N_BLOCK="${2:-32}"

$PY -m pytest -q tests_text --ignore=tests_text/test_ci_text_stack_smoke_script.py -k "not ci_text_stack_smoke_script"
bash tools/run_all_text_demos.sh "$SEED" "$N_BLOCK"
