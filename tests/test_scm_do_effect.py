from vsc_repo.scm import BinarySCM, SCMParams, p_y1_given_x, backdoor_adjustment_p_y1_do_x

def test_backdoor_matches_true_do_effect_and_naive_fails():
    scm = BinarySCM(SCMParams())
    obs = scm.sample_observational(n=80000, seed=0)
    true0 = scm.true_p_y1_do_x(0, n_mc=200000, seed=1)
    true1 = scm.true_p_y1_do_x(1, n_mc=200000, seed=2)
    naive0 = p_y1_given_x(obs, 0)
    naive1 = p_y1_given_x(obs, 1)
    bd0 = backdoor_adjustment_p_y1_do_x(obs, 0)
    bd1 = backdoor_adjustment_p_y1_do_x(obs, 1)

    assert max(abs(bd0-true0), abs(bd1-true1)) < 0.03
    assert max(abs(naive0-true0), abs(naive1-true1)) > 0.08
