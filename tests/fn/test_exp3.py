"""Tests for exp3.exp3_bandit.

Anchors: exact hand arithmetic of the Exp3 update (Auer et al 2002,
figure 1) -- uniform first-round probabilities, the one-step weight
update recomputed independently, and the gamma = 1 uniform limit.
"""

import math

from morie.fn import _array_core as np
from morie.fn.exp3 import exp3


def test_exp3_first_round_probs_are_uniform_mixture():
    # w_i(1) = 1 for all i, so p_i(1) = (1-g)/K + g/K = 1/K exactly.
    x = [[0.2, 0.9, 0.4]] * 3
    r = exp3(x, 0.3, seed=0)
    for j in range(3):
        assert abs(r["probs"][0, j] - 1.0 / 3.0) < 1e-16


def test_exp3_one_step_weight_hand_update():
    g = 0.5
    x = [[0.8, 0.3]]
    r = exp3(x, g, seed=4)
    i = int(r["actions"][0])
    # replay: p_i(1) = 1/2; w_i(2) = exp(g * (x_i / 0.5) / 2)
    expected = math.exp(g * (x[0][i] / 0.5) / 2.0)
    assert abs(r["weights"][i] - expected) < 1e-15
    other = 1 - i
    assert r["weights"][other] == 1.0


def test_exp3_gamma_one_stays_uniform():
    # gamma = 1: p_i(t) = 1/K for every t regardless of the weights.
    rng = np.random.default_rng(2)
    x = [[float(rng.uniform()) for _ in range(4)] for _ in range(12)]
    r = exp3(x, 1.0, seed=8)
    for t in range(12):
        for j in range(4):
            assert abs(r["probs"][t, j] - 0.25) < 1e-16


def test_exp3_favors_dominant_arm():
    x = [[1.0, 0.0]] * 200
    r = exp3(x, 0.2, seed=1)
    assert r["estimate"] == 0.0
    assert r["weights"][0] > r["weights"][1]
    # arm 0 must be played much more often than arm 1
    n0 = sum(1 for a in r["actions"] if a == 0.0)
    assert n0 > 140


def test_exp3_rejects_bad_gamma():
    try:
        exp3([[0.5, 0.5]], 0.0)
    except ValueError:
        return
    raise AssertionError("gamma_ = 0 accepted")
