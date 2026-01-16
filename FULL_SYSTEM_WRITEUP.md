# `vsc_world_causal_planner_v1` — Full System Write-Up (Deterministic Causal + World-Model + Planning + Safety Stack)

## 0) What this repository is

`vsc_world_causal_planner_v1` is a **deterministic, test-backed causal identification + world-model learning + planning + safety gating stack**, implemented as a **reversible text world**.

This repo does **not** claim:

- “it talks about causality”
- “it feels like an agent”
- “it generates plausible text”

This repo **proves**, with **numerical inequalities**, **PASS lines**, and **written JSON artifacts**, that the following pipeline runs end-to-end:

\[
\text{SCM Identification}
\Rightarrow
\widehat{T}(s'|s,a)
\Rightarrow
\pi^* \text{ via Value Iteration}
\Rightarrow
\epsilon\text{-risk constrained selection}
\Rightarrow
\text{contract output + rejected counterfactual}
\]

Everything is reproducible because:

- randomness is seeded,
- outputs are written to JSON,
- tests assert fixed thresholds and invariants.

This is a **checkable backend kernel** that can later be wrapped with richer interfaces **without weakening the guarantees**.

---

## 1) What the repo proves (four core subsystems)

### 1.1 Causal identification (SCM backdoor vs do-truth)

The repo contains an SCM that generates **confounded observational data** \((Z,X,Y)\) where naive correlation fails.

The system computes:

- **Naive estimate:** \(\widehat{P}(Y|X)\)
- **Backdoor estimate:** \(\widehat{P}(Y|do(X))\) computed from observational counts via stratifying over \(Z\)
- **Interventional truth:** \(P(Y|do(X))\) estimated from explicit do-sampling

The demo enforces falsifiable inequalities:

**Backdoor matches interventional truth**
\[
\max_x
\left|
\widehat{P}(Y|do(x))_{\text{backdoor}}
-
P(Y|do(x))_{\text{interventional}}
\right|
\le
\text{tol}
\]

**Naive fails under confounding**
\[
\min_x
\left|
\widehat{P}(Y|x)_{\text{naive}}
-
P(Y|do(x))
\right|
\ge
0.07
\]

PASS line form:

```
[PASS] TEXT_SCM_BACKDOOR: max_abs_err_backdoor=... tol=... min_gap_naive=...
```

Meaning (precisely):

- naive correlation is **provably wrong**
- backdoor adjustment recovers the **correct do-effect** inside tolerance

---

### 1.2 World model learning (MLE transition kernel validated by mean L1)

The repo’s “world” is a discrete state machine with actions.
The simulator defines the true transition kernel:

\[
T(s' \mid s, a)
\]

The repo learns an empirical MLE model:

\[
\widehat{T}(s' \mid s,a)
\]

and validates it numerically by mean \(L^1\) distance:

\[
\mathbb{E}_{(s,a)}
\left[
\left\|T(\cdot|s,a)-\widehat{T}(\cdot|s,a)\right\|_1
\right]
\le
\tau
\]

PASS line form:

```
[PASS] TEXT_WORLD_MODEL_TRANSITION_L1: mean_l1=... threshold=...
```

For larger text worlds (paragraph/document/block), evaluation uses anchor sampling and repeats per key to keep measurement finite — but the invariant is unchanged:

> learned transition probabilities must remain inside a fixed error budget.

---

### 1.3 Planning correctness (Value Iteration matches brute force)

The repo computes an optimal plan in two independent ways:

1. Value Iteration (DP)
2. Brute force enumeration of stationary policies (gold baseline)

It asserts exact agreement:

\[
|V_{\text{VI}} - V_{\text{BF}}| \le 10^{-6}
\]

PASS line form:

```
[PASS] TEXT_PLANNING_VI_EQUALS_BRUTE_FORCE: abs_return_diff=0.000000 ...
```

Meaning:

- the VI implementation is not “trusted”
- it is **pinned to an exhaustive solver**

---

### 1.4 Safety constraint (ε gating under hard risk bound)

Safety is a **constrained optimizer**, not a slogan.

- objective: maximize return
- constraint: risk ≤ ε

\[
\max_{\pi} \;\mathbb{E}[R|\pi]
\quad
\text{s.t.}
\quad
\text{Risk}(\pi)\le \epsilon
\]

The demos construct explicit cases where the unconstrained optimum violates ε, forcing feasibility behavior.

PASS line form:

```
[PASS] TEXT_SAFETY_CONSTRAINT_POLICY_SELECTED: chosen_risk=0.0000 epsilon=0.12 opt_risk=0.1800
```

Meaning (exact):

- optimal unconstrained action is unsafe: \(\text{opt\_risk} > \epsilon\)
- chosen action satisfies the safety bound: \(\text{chosen\_risk} \le \epsilon\)
- selector mode reports whether it chose “best feasible” or a forced tradeoff

---

## 2) Why text matters here: reversible state machine, not text generation

The text layer is not decoration.

The repo uses text as a **lossless interface to discrete latent state**.

Core invariant:

\[
\text{parse}(\text{render}(s)) = s
\]

There is a **finite enumerated state space** (currently **256 states**) and deterministic render/parse functions.

So “text world” means:

- discrete state \(s\) → rendered to text
- that text → parsed back to the same \(s\)
- planning/safety runs on \(s\), not on “string vibes”

### Why “the claim is unspecified” appears
That repetition is a deliberate renderer placeholder for under-specified fact masks.

It is **not hallucination**.
It marks states whose fact mask does not produce a concrete surface claim.

---

## 3) Artifacts and proofs: JSON as evidence, not logging

Every meaningful run produces:

- a JSON artifact containing metrics, decisions, thresholds, and rejected branches
- a PASS line summarizing the verified condition

This repo treats JSON as a **proof-carrying run record**:

- tests load JSON and assert inequalities
- humans inspect evidence directly
- runs reproduce from seeds

Examples of smoke outputs:

- `results/text_sentence_demo.json`
- `results/text_paragraph_world_demo.json`
- `results/text_document_tradeoff_demo.json`
- `results/text_block_world_demo.json`
- `results/text_full_stack_demo.json`

The smoke script is therefore a **system proof**:
unit tests + integration demos + artifact writes.

---

## 4) Macro-beam safety gating is a pinned regression anchor

Block-world beam search has explicit semantics:

- **risk** is path maximum (not summed)
- **return** is summed

So safety means:

> never exceed ε at any step.

The macro-beam regression pins a step-function property of ε:

- total candidates: 374,544
- at ε = 0.99: reject 53,429; keep 321,115
- at ε = 1.00: reject 0; keep 374,544
- every rejected branch has risk_max_if_taken = 1.0

Written to JSON as:

```
BLOCK_MACRO_BEAM_SEARCH.{n_candidates_total, n_kept_total, n_rejected_total, rejected_counterfactuals}
```

Meaning:

- ε is a **hard cutoff**, not a soft preference function.

---

## 5) Latest upgrade: enhanced sentence layer is first-class and still reversible

The repo includes an enhanced surface text system:

- 5–10× larger vocabulary
- light syntactic sugar
- controlled “strength/confidence” phrasing

Crucially:

✅ semantic identity stays pinned to the same discrete `sid`
✅ enhanced text still round-trips to the same `sid` for all 256 states

So this is the correct bridge to “chat-like surface” **without breaking the kernel**:

- naturalness rises
- invertibility remains exact

Strength/confidence is treated correctly as **surface force** (expression only).
It never rewrites underlying world truth.

---

# Training: Neural inverse map (text → `sid`) is now proven working

## 6) What training is doing

The repo is training a supervised multi-class classifier:

\[
f_\theta(\text{text}) \rightarrow \text{sid} \in \{0,\dots,255\}
\]

This does **not** replace symbolic truth.
It is a learned perception module, which can be compared against the guaranteed symbolic parser.

---

## 6.1 Training configuration (exact)

Script: `experiments/train_neural_parser.py`

Run config:

- Model: `distilbert-base-uncased`
- Labels: 256 classes
- Train set: 1,000,000 JSONL examples
- Test set: 50,000 JSONL examples
- Batch size: train 64, eval 128
- Max token length: 96
- Epochs: 1
- Logging every 200 steps
- Eval every 1000 steps
- Save checkpoints every 1000 steps

Total steps:

\[
\frac{1{,}000{,}000}{64} = 15{,}625
\]

---

## 6.2 Final observed result (completed run)

Training finished at step **15,625 / 15,625 (100%)**.

Final metrics:

```
[PASS] TRAIN_NEURAL_PARSER:
  eval_loss = 0.042502064257860184
  eval_acc  = 0.96912
  epoch     = 1.0
```

Meaning:

- the neural inverse map decodes the 256-state world
- at **~96.9% accuracy** after one epoch
- and the run is fully reproducible from seed 0 + written checkpoints

---

## 6.3 Why training looks “slow” / why ETA spikes

The cause:

- evaluation runs every 1000 steps and takes minutes on MPS
- checkpoint saving adds pauses
- tqdm smears pauses into per-step ETA

The run is not stalled — it is paying the intended eval/save cost.

---

## 6.4 The warnings are non-fatal

- LibreSSL / urllib3 warning: environment noise
- HF tokenizer deprecation warning: API noise
- pin_memory warning on MPS: normal

None of these impact correctness of training.

---

# Next implementation: `text_world/agent/`

## 7) Next subsystem: deterministic dialogue layer (not an LLM)

Add:

```
text_world/agent/
  dialogue_state.py
  intent_schema.py
  speaker_surface.py
  chat_loop.py
```

This does **not** attempt to generate an LLM.

It provides:

- a dialogue state machine
- an intent object (proof-carrying “how to speak” contract)
- a deterministic speaker surface (controlled variation without drift)
- an interactive loop that:
  - reads user input
  - attempts symbolic parse into `sid`
  - produces a deterministic main reply
  - emits a safety verdict + rejected counterfactual block
  - records full dialogue history deterministically

### 7.1 `dialogue_state.py`
Minimal memory structure:

- turn index
- user text
- parsed `sid_in` (if available)
- resulting `sid_out`
- chosen action (placeholder first)
- `risk_max`, `epsilon`

This is the minimal “conversation memory” needed to turn planning into dialogue.

### 7.2 `intent_schema.py`
Defines `SpeakIntent` (pinned surface contract):

- `sid`
- tone
- strength
- goal
- include/exclude constraints
- metadata for traceability

Serializable to JSON.

### 7.3 `speaker_surface.py`
Deterministic surface generator:

- takes base text + `SpeakIntent`
- produces fluent variants
- uses seeded RNG derived from `(seed, turn, sid)`
- never introduces nondeterminism

Variation without semantic drift.

### 7.4 `chat_loop.py`
Minimal interactive interface that prints proof blocks:

```
MAIN_REPLY:
...

SAFETY_VERDICT:
{...}

COUNTERFACTUAL_REJECTED_BRANCH:
{...}
```

Phase 1: `_select_next_sid` stub (identity).
Phase 2: wire to safe planner + ε-gate transition step.

---

# Repo status summary (right now)

## Complete and proven
- SCM identification vs do-truth
- transition learning with L1 validation
- VI correctness vs brute force
- ε-risk constrained selection
- block macro-beam safety gating regression anchor
- reversible clean text world
- reversible enhanced text world (256 states)
- dataset generation + verification tooling
- neural parser training pipeline **completed**
- neural parser achieved **eval_acc = 0.96912** at epoch 1

## What is running now
- neural run completed
- latest checkpoint + model saved to `models/neural_parser_resume`

## What we implement next
- deterministic `text_world/agent/` chat interface layer
- wire chat loop into real planner + ε-gate transition step

---

# Environment bootstrap (shell commands)

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If Preview is frozen:

```bash
killall Preview
```

---

# One code-level cleanup (canonical)

In `train_neural_parser.py`, remove the duplicate RESUME logic:

```py
resume = os.environ.get("RESUME")
load_dir = resume if resume else model_name
```

(Behavior unchanged; just canonical cleanup.)
