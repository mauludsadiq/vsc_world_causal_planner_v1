from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class StepOut:
    sid_out: int
    chosen_action: int
    chosen_risk: float
    mode: str
    rejected: Dict[str, Any]


def _load_callable(module_name: str, symbol: str) -> Optional[Callable[..., Any]]:
    try:
        m = importlib.import_module(module_name)
    except Exception:
        return None
    fn = getattr(m, symbol, None)
    return fn if callable(fn) else None


def _first_available(cands: Tuple[Tuple[str, str], ...]) -> Optional[Callable[..., Any]]:
    for mod, sym in cands:
        fn = _load_callable(mod, sym)
        if fn is not None:
            return fn
    return None


def _build_world() -> Tuple[Any, Any]:
    """
    Build a world object plus its env module.
    We prefer SentenceWorld (smallest) and use env_paragraph sampling.
    """
    last_err: Optional[Exception] = None

    # 1) Sentence world (planning_text expects this shape)
    try:
        ep = importlib.import_module("text_world.env_paragraph")
        sw = getattr(ep, "build_sentence_world", None)
        if callable(sw):
            world = sw()
            return ep, world
    except Exception as e:
        last_err = e

    # 2) Paragraph world
    try:
        ep = importlib.import_module("text_world.env_paragraph")
        pw = getattr(ep, "build_paragraph_world", None)
        if callable(pw):
            world = pw()
            return ep, world
    except Exception as e:
        last_err = e

    # 3) Block world (needs n)
    try:
        eb = importlib.import_module("text_world.env_block")
        bw = getattr(eb, "build_block_world", None)
        if callable(bw):
            world = bw(4)
            return eb, world
    except Exception as e:
        last_err = e

    # 4) Document world
    try:
        ed = importlib.import_module("text_world.env_document")
        dw = getattr(ed, "build_document_world", None)
        if callable(dw):
            world = dw()
            return ed, world
    except Exception as e:
        last_err = e

    raise RuntimeError(f"planner_bridge: could not build a world (Sentence/Paragraph/Block/Document). last_err={last_err}")


def _select_policy_under_risk(world: Any, s0: int, H: int, epsilon: float) -> Dict[str, Any]:
    """
    Uses the same selector used in the demos:
      text_world.planning_text.select_policy_under_risk(world, candidates, s0, H, epsilon)

    Note: no seed kwarg.
    candidates := world.actions if present else range(n_actions)
    """
    sel = _first_available((("text_world.planning_text", "select_policy_under_risk"),))
    if sel is None:
        raise RuntimeError("planner_bridge: could not import text_world.planning_text.select_policy_under_risk")

    # candidates
    candidates: List[int]
    if hasattr(world, "actions"):
        candidates = list(getattr(world, "actions"))
    else:
        # fallback: infer number of actions from transition table keys if possible
        if hasattr(world, "T"):
            T = getattr(world, "T")
            acts = sorted({int(a) for (_, a) in T.keys()})
            candidates = acts
        elif hasattr(world, "_T"):
            T = getattr(world, "_T")
            acts = sorted({int(a) for (_, a) in T.keys()})
            candidates = acts
        else:
            raise RuntimeError("planner_bridge: world has no .actions and no (T/_T) transition table to infer actions")
    # planning_text expects candidate POLICIES, not candidate ACTIONS.
    # For H=1, a one-step policy dict {s0: action} is sufficient.
    action_ids = list(candidates)
    candidates = [{int(s0): int(a)} for a in action_ids]


    # Call with positional args to avoid signature drift
    return sel(world, candidates, int(s0), int(H), float(epsilon))


def _sample_transition(env_mod: Any, world: Any, s: int, act: int, seed: int) -> int:
    """
    env_paragraph.sample_transition expects a world with _T table and typically
    takes either:
      (world, s, act) or (world, s, act, rng)

    IMPORTANT:
      Some envs use rng.random() so the 4th arg must be an RNG object,
      not an int seed.
    """
    fn = getattr(env_mod, "sample_transition", None)
    if not callable(fn):
        raise RuntimeError(f"planner_bridge: env module {env_mod.__name__} has no sample_transition")

    class _EnvWorldShim:
        def __init__(self, w: Any):
            self._w = w
            if hasattr(w, "_T"):
                self._T = getattr(w, "_T")
            elif hasattr(w, "T"):
                self._T = getattr(w, "T")
            else:
                self._T = None

        def __getattr__(self, name: str) -> Any:
            return getattr(self._w, name)

    w_for_env = world
    if not hasattr(world, "_T") and hasattr(world, "T"):
        w_for_env = _EnvWorldShim(world)

    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        n = len(params)
    except Exception:
        params = []
        n = 0

    if n == 3:
        return int(fn(w_for_env, int(s), int(act)))

    if n == 4:
        # Most common: 4th param is an RNG (has .random()).
        import random
        rng = random.Random(int(seed))

        # If we can see the param name, use it as a strong hint.
        p4 = params[3].name.lower() if len(params) >= 4 else ""
        if "rng" in p4 or "random" in p4:
            return int(fn(w_for_env, int(s), int(act), rng))

        # Otherwise: try RNG first, then fall back to seed-as-int.
        try:
            return int(fn(w_for_env, int(s), int(act), rng))
        except Exception:
            return int(fn(w_for_env, int(s), int(act), int(seed)))

    # Unknown arity: try keyword seed
    try:
        return int(fn(w_for_env, int(s), int(act), seed=int(seed)))
    except Exception as e:
        raise RuntimeError(f"planner_bridge: could not call sample_transition. fn={fn} err={e}")

def step_safe(sid_in: int, epsilon: float, seed: int) -> StepOut:
    """
    Single safe step:
      1) build world
      2) select policy under risk (same contract used in demos)
      3) take one sampled transition using env's sample_transition
      4) return StepOut
    """
    env_mod, world = _build_world()

    selection = _select_policy_under_risk(world, s0=int(sid_in), H=1, epsilon=float(epsilon))

    chosen_action = int(selection["chosen_action"])
    chosen_risk = float(selection["chosen_risk"])
    mode = str(selection.get("mode", "feasible_best_return"))

    rejected = selection.get("rejected", None)
    if not isinstance(rejected, dict):
        rejected = {"reason": "risk bound", "epsilon": float(epsilon), "opt_risk": float(selection.get("opt_risk", chosen_risk))}

    if chosen_risk > float(epsilon):
        raise RuntimeError(f"planner_bridge.step_safe: unsafe choice chosen_risk={chosen_risk} epsilon={epsilon}")

    sid_out = _sample_transition(env_mod, world, int(sid_in), int(chosen_action), int(seed))

    return StepOut(
        sid_out=int(sid_out),
        chosen_action=int(chosen_action),
        chosen_risk=float(chosen_risk),
        mode=mode,
        rejected=dict(rejected),
    )
