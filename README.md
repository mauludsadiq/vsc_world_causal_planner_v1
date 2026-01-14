# World Modeling + Planning (test-backed repo identity)

This repo is a deterministic harness that validates three linked capabilities in a small controlled environment:

1) **Causal estimation under confounding**: backdoor adjustment matches the true do-effect; naive conditioning deviates.
2) **World-model learning**: a learned controlled transition model matches ground-truth dynamics in L1.
3) **Planning correctness**: value iteration returns the same optimal stationary policy as brute-force search.

## What “tests green” means (exact PASS lines)

```
[PASS] SCM_DO_EFFECT_BACKDOOR: backdoor_x0=0.307904 backdoor_x1=0.777974 max_abs_err_backdoor=0.00443 max_abs_gap_naive=0.090943 min_gap_naive=0.08 naive_x0=0.218209 naive_x1=0.845511 tol=0.03 true_do_x0=0.309152 true_do_x1=0.773544 :: Backdoor matches do-effect; naive conditional deviates under confounding.
[PASS] WORLD_MODEL_TRANSITION_L1: mean_l1=0.001385 samples=32000 threshold=0.06 :: Learned controlled transition model approximates ground truth.
[PASS] PLANNING_VI_EQUALS_BRUTE_FORCE: abs_return_diff=0.0 bf_policy=[0, 0, 0] bf_return=4.0 vi_policy=[0, 0, 0] vi_return=4.0 :: Value iteration matches brute-force optimal stationary policy.
```

## Setup

### Requirements
- Python 3.9+ (3.10+ also fine)
- `pip`

### Create venv + install
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

### Run the test suite
```bash
pytest -q
```

### Reproduce the PASS transcript (recommended)
Run tests with minimal output (PASS lines come from the tests):
```bash
pytest -q
```

## Contract

If the suite is green, the repo guarantees (in this harness):
- Backdoor-adjusted causal estimates match the ground-truth do-effect within tolerance.
- The learned transition model matches ground-truth transitions (mean L1 below threshold).
- Value iteration’s policy/return match brute-force optimal planning exactly for the tested instance.
