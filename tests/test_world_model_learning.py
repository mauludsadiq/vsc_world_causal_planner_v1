from vsc_repo.run import build_demo_env
from vsc_repo.mdp import collect_rollouts, learn_model_from_transitions, l1_transition_error

def test_transition_model_learning_is_close():
    env = build_demo_env()
    trans = collect_rollouts(env, n_rollouts=3000, horizon=10, seed=0)
    T_hat, _ = learn_model_from_transitions(trans, nS=env.nS, nA=env.nA, laplace=1.0)
    l1 = l1_transition_error(env.T, T_hat)
    assert l1 < 0.06
