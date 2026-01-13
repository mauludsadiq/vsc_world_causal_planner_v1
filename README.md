# vsc_world_causal_planner

Operational repo that forces four capabilities to exist and be testable:

1. **Causal effect under intervention** (SCM + do-operator)
2. **World model** (learned controlled transition model)
3. **Planning** (value iteration vs brute-force optimality)
4. **Safety constraint** (policy selection under risk bound)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -q
python -m vsc_repo.run --seed 0
```

## Output

Runner prints:

`[PASS] <CHECK_NAME>: key=value ...`

and writes `artifacts/report.json`.
