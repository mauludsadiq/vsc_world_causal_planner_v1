from __future__ import annotations

import argparse
import os
import numpy as np

from vsc_repo.report import Reporter
from vsc_repo.scm import BinarySCM, SCMParams, p_y1_given_x, backdoor_adjustment_p_y1_do_x
from vsc_repo.mdp import MDPEnv, collect_rollouts, learn_model_from_transitions, l1_transition_error
from vsc_repo.planner import value_iteration, brute_force_optimal_policy, eval_policy_exact
from vsc_repo.constraints import estimate_risk_mc, constrained_select

def build_demo_env() -> MDPEnv:
    T = np.array([
        [[0.85, 0.15, 0.00], [0.55, 0.35, 0.10]],
        [[0.10, 0.90, 0.00], [0.05, 0.70, 0.25]],
        [[0.00, 0.00, 1.00], [0.00, 0.00, 1.00]],
    ], dtype=float)
    R = np.array([[0.2, 1.0], [0.2, 1.2], [-2.0, -2.0]], dtype=float)
    start = np.array([0.5, 0.5, 0.0], dtype=float)
    return MDPEnv(T=T, R=R, harm_states=(2,), start_dist=start)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rep = Reporter()

    # SCM check
    scm = BinarySCM(SCMParams())
    obs = scm.sample_observational(n=60000, seed=args.seed)
    true0 = scm.true_p_y1_do_x(0, n_mc=250000, seed=args.seed+1)
    true1 = scm.true_p_y1_do_x(1, n_mc=250000, seed=args.seed+2)
    naive0 = p_y1_given_x(obs, 0)
    naive1 = p_y1_given_x(obs, 1)
    bd0 = backdoor_adjustment_p_y1_do_x(obs, 0)
    bd1 = backdoor_adjustment_p_y1_do_x(obs, 1)

    err_bd = max(abs(bd0-true0), abs(bd1-true1))
    gap_naive = max(abs(naive0-true0), abs(naive1-true1))

    rep.add(
        "SCM_DO_EFFECT_BACKDOOR",
        passed=(err_bd < 0.03 and gap_naive > 0.08),
        metrics={
            "true_do_x0": round(true0, 6),
            "true_do_x1": round(true1, 6),
            "naive_x0": round(naive0, 6),
            "naive_x1": round(naive1, 6),
            "backdoor_x0": round(bd0, 6),
            "backdoor_x1": round(bd1, 6),
            "max_abs_err_backdoor": round(err_bd, 6),
            "max_abs_gap_naive": round(gap_naive, 6),
            "tol": 0.03,
            "min_gap_naive": 0.08,
        },
        message="Backdoor matches do-effect; naive conditional deviates under confounding.",
    )

    # World model learning
    env = build_demo_env()
    trans = collect_rollouts(env, n_rollouts=4000, horizon=8, seed=args.seed)
    T_hat, R_hat = learn_model_from_transitions(trans, nS=env.nS, nA=env.nA, laplace=1.0)
    l1 = l1_transition_error(env.T, T_hat)
    rep.add(
        "WORLD_MODEL_TRANSITION_L1",
        passed=(l1 < 0.06),
        metrics={"mean_l1": round(l1, 6), "threshold": 0.06, "samples": int(len(trans["s"]))},
        message="Learned controlled transition model approximates ground truth.",
    )

    # Planning
    vi = value_iteration(env.T, env.R, gamma=0.95, tol=1e-12)
    bf_pi, bf_J = brute_force_optimal_policy(env.T, env.R, gamma=0.95, start_dist=env.start_dist)
    vi_J = eval_policy_exact(env.T, env.R, vi.policy, gamma=0.95, start_dist=env.start_dist)
    same = bool(np.array_equal(vi.policy, bf_pi))
    rep.add(
        "PLANNING_VI_EQUALS_BRUTE_FORCE",
        passed=(same and abs(vi_J - bf_J) < 1e-8),
        metrics={
            "vi_policy": vi.policy.tolist(),
            "bf_policy": bf_pi.tolist(),
            "vi_return": round(vi_J, 10),
            "bf_return": round(bf_J, 10),
            "abs_return_diff": float(abs(vi_J - bf_J)),
        },
        message="Value iteration matches brute-force optimal stationary policy.",
    )

    # Safety constraint selection
    pi_opt = bf_pi
    pi_safe = np.zeros(env.nS, dtype=int)
    cand = [pi_opt, pi_safe]
    rets = [
        eval_policy_exact(env.T, env.R, pi_opt, gamma=0.95, start_dist=env.start_dist),
        eval_policy_exact(env.T, env.R, pi_safe, gamma=0.95, start_dist=env.start_dist),
    ]
    risks = [
        estimate_risk_mc(env.T, pi_opt, env.harm_states, env.start_dist, horizon=25, n_mc=15000, seed=args.seed+10),
        estimate_risk_mc(env.T, pi_safe, env.harm_states, env.start_dist, horizon=25, n_mc=15000, seed=args.seed+11),
    ]
    eps = 0.12
    chosen = constrained_select(cand, rets, risks, epsilon=eps)
    rep.add(
        "SAFETY_CONSTRAINT_POLICY_SELECTED",
        passed=(chosen.chosen_risk <= eps + 1e-9),
        metrics={
            "epsilon": eps,
            "risk_opt": round(risks[0], 6),
            "risk_safe": round(risks[1], 6),
            "ret_opt": round(rets[0], 6),
            "ret_safe": round(rets[1], 6),
            "chosen_policy": chosen.chosen_policy.tolist(),
            "chosen_risk": round(chosen.chosen_risk, 6),
            "chosen_return": round(chosen.chosen_return, 6),
        },
        message="Selected a policy satisfying risk bound (or minimum-risk fallback).",
    )

    rep.print()
    rep.write_json(os.path.join("artifacts", "report.json"))
    return 0 if rep.summary()["all_passed"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
