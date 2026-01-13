from __future__ import annotations
import random
from text_world.scm_text import (
    TextSCMParams,
    sample_observational,
    sample_interventional_doX,
    estimate_naive_p_y_given_x,
    backdoor_do_effect,
    true_do_effect_from_interventional,
)

def test_text_scm_backdoor_matches_do_and_naive_fails():
    seed = 0
    p = TextSCMParams()
    obs = sample_observational(random.Random(seed), n=200000, p=p)
    do0 = sample_interventional_doX(random.Random(seed + 1), n=200000, x_do=0, p=p)
    do1 = sample_interventional_doX(random.Random(seed + 2), n=200000, x_do=1, p=p)

    naive = estimate_naive_p_y_given_x(obs)
    backdoor = backdoor_do_effect(obs)
    true_do = {0: true_do_effect_from_interventional(do0), 1: true_do_effect_from_interventional(do1)}

    tol = 0.03
    assert max(abs(backdoor[x] - true_do[x]) for x in (0,1)) <= tol
    assert min(abs(naive[x] - true_do[x]) for x in (0,1)) >= 0.07
