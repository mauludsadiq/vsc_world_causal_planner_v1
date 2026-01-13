import numpy as np
from vsc_repo.run import build_demo_env
from vsc_repo.planner import value_iteration, brute_force_optimal_policy, eval_policy_exact

def test_value_iteration_matches_bruteforce_policy():
    env = build_demo_env()
    vi = value_iteration(env.T, env.R, gamma=0.95, tol=1e-12)
    bf_pi, bf_J = brute_force_optimal_policy(env.T, env.R, gamma=0.95, start_dist=env.start_dist)
    vi_J = eval_policy_exact(env.T, env.R, vi.policy, gamma=0.95, start_dist=env.start_dist)
    assert np.array_equal(vi.policy, bf_pi)
    assert abs(vi_J - bf_J) < 1e-8
