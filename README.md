vsc_world_causal_planner_v1 — Deterministic World Modeling + Planning + Proof-Carrying Agent

0) Repo identity

vsc_world_causal_planner_v1 is a deterministic, test-backed causal + world-model + planning + safety + agent harness implemented as a small reversible text world.

This repo does not claim scale, intelligence, or general “chat ability.”

It proves—via numerical inequalities, PASS lines, and cryptographically anchored JSON artifacts—that the following end-to-end chain works:

SCM identification → backdoor do-effect → learned controlled transition model → optimal planning (VI) → epsilon-risk safety selection → proof-carrying dialogue artifacts (sha256)

Truth in this repo means:

A claim is valid only if it produces a deterministic witness artifact that passes tests.

1) What the repo proves

1.1 Causal estimation under confounding (Backdoor)

The repo contains a small SCM with confounding and proves:

- Backdoor adjustment matches the true do-effect within tolerance
- Naive conditioning deviates under confounding

Backdoor computes:

E[Y | do(X=x)] = sum_z E[Y | X=x, Z=z] P(Z=z)

1.2 World-model learning (controlled transition model)

The repo learns a controlled transition model T_hat(s' | s, a) and proves mean L1 error to ground truth is below threshold.

1.3 Planning correctness (Value Iteration equals brute force)

The repo proves that value iteration returns the same optimal stationary policy and return as brute-force search on the tested instance.

1.4 Safety-constrained selection (epsilon-risk)

The repo enforces an epsilon-risk constraint and deterministically selects the best feasible action, recording the verdict in artifacts.

2) Agent layer + proof-carrying dialogue artifacts

2.1 Inverse decoding (symbolic-first → neural fallback)

User text is decoded into a discrete state id sid_hat.

- Symbolic-first parse is attempted first (clean grammar)
- Neural fallback is used only when confidence gating passes

Confidence gating is enforced using thresholds tau_p and tau_margin:

p_top1 >= tau_p
(p_top1 - p_top2) >= tau_margin

Every decode produces a proof object recording mode, sid_hat, probabilities, margin, entropy, thresholds, and seed.

2.2 Proof-carrying dialogue turns

Each agent run emits turn-by-turn witness records that include:

turn, user_text, decode, sid_in, sid_out, assistant_text, main_reply, safety_verdict, rejected_counterfactuals

The full dialogue artifact is stable JSON and includes a sha256 commitment.

3) What “tests green” means

When the suite is green, the repo has proven (in this harness):

- Backdoor do-effect matches ground truth within tolerance
- Learned transition model matches ground truth under mean L1 threshold
- Value iteration matches brute-force optimal policy/return exactly
- Safety constraint selection is deterministic and recorded
- Agent loop emits proof-carrying dialogue artifacts with sha256
- Neural inverse decode contract is deterministic and confidence-gated

4) Setup

Requirements:
- Python 3.9+
- pip

Create venv and install:

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

5) Run

Run full test suite:

python -m pytest -q

Run text-world causal + planning demos (write JSON artifacts):

python -m experiments.text_sentence_demo results/text_sentence_demo.json 0
python -m experiments.text_paragraph_world_demo results/text_paragraph_world_demo.json 0

Run agent chat demo (symbolic-only):

python -m experiments.agent_chat_demo results/agent/agent_chat_demo.json 0

Run agent chat demo (symbolic-first + neural fallback):

python -m experiments.agent_chat_demo results/agent/agent_chat_demo_with_neural.json 0 models/neural_parser_resume

Run CI-style smoke script:

bash tools/ci_text_stack_smoke.sh 0 8

6) Contract

If the suite is green, this repo guarantees:

- Backdoor-adjusted causal estimates match ground-truth do-effect within tolerance
- Learned controlled transition model matches ground truth under mean L1 threshold
- Value iteration matches brute-force optimal planning on the tested instance
- Epsilon-risk safety decision is deterministic and recorded
- Proof-carrying dialogue artifacts are schema-valid and sha256-anchored
