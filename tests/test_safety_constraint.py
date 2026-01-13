import numpy as np
from vsc_repo.run import build_demo_env
from vsc_repo.planner import brute_force_optimal_policy, eval_policy_exact
from vsc_repo.constraints import estimate_risk_mc, constrained_select

def test_constrained_selection_satisfies_risk_bound():
    env = build_demo_env()
    pi_opt, _ = brute_force_optimal_policy(env.T, env.R, gamma=0.95, start_dist=env.start_dist)
    pi_safe = np.zeros(env.nS, dtype=int)

    cand = [pi_opt, pi_safe]
    rets = [
        eval_policy_exact(env.T, env.R, pi_opt, gamma=0.95, start_dist=env.start_dist),
        eval_policy_exact(env.T, env.R, pi_safe, gamma=0.95, start_dist=env.start_dist),
    ]
    risks = [
        estimate_risk_mc(env.T, pi_opt, env.harm_states, env.start_dist, horizon=25, n_mc=12000, seed=0),
        estimate_risk_mc(env.T, pi_safe, env.harm_states, env.start_dist, horizon=25, n_mc=12000, seed=1),
    ]
    eps = 0.12
    chosen = constrained_select(cand, rets, risks, epsilon=eps)
    assert chosen.chosen_risk <= eps + 1e-9
