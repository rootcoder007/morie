"""rng142: Wiener cross-correlation vector (Rangayyan 2024, Eq. 3.160/3.161, p. 174)."""

import numpy as np
import pytest

from morie.fn.rng142 import rangayyan_ch3_cross_correlation_vector as theta_vec


def test_rng142_matches_the_definition_computed_by_hand():
    """theta(-k) = E[x(n-k) d(n)] over the n where the tap vector exists."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = np.array([2.0, 0.0, 1.0, -1.0, 3.0])
    M = 3
    got = theta_vec(x, d, M)["array"]
    # n runs 2..4 (0-based), so d[2:] = (1, -1, 3).
    #   k=0: x[2:5]*d[2:] = (3,4,5)*(1,-1,3) -> (3, -4, 15), mean 14/3
    #   k=1: x[1:4]*d[2:] = (2,3,4)*(1,-1,3) -> (2, -3, 12), mean 11/3
    #   k=2: x[0:3]*d[2:] = (1,2,3)*(1,-1,3) -> (1, -2,  9), mean  8/3
    assert got == pytest.approx([14 / 3, 11 / 3, 8 / 3])


def test_rng142_delta_desired_response_selects_one_lag():
    """With d a unit impulse at n0, Theta reads off x(n0-k)/(N-M+1) directly."""
    x = np.arange(1.0, 9.0)
    d = np.zeros(8)
    d[5] = 1.0
    M = 3
    got = theta_vec(x, d, M)["array"]
    n_avg = 8 - M + 1
    assert got == pytest.approx([x[5] / n_avg, x[4] / n_avg, x[3] / n_avg])


def test_rng142_length_is_the_filter_length():
    rng = np.random.default_rng(1)
    r = theta_vec(rng.standard_normal(64), rng.standard_normal(64), 5)
    assert r["array"].shape == (5,)
    assert r["M"] == 5
    assert r["n"] == 64


def test_rng142_uncorrelated_signals_give_approximately_zero():
    rng = np.random.default_rng(99)
    n = 200_000
    got = theta_vec(rng.standard_normal(n), rng.standard_normal(n), 4)["array"]
    assert np.all(np.abs(got) < 0.02)


def test_rng142_rejects_mismatched_lengths():
    """d is the desired-response SIGNAL; the generated test passed d=5."""
    with pytest.raises(ValueError, match="same length"):
        theta_vec(np.zeros(10), np.zeros(9), 3)


def test_rng142_rejects_bad_filter_length():
    with pytest.raises(ValueError, match=">= 1"):
        theta_vec(np.zeros(10), np.zeros(10), 0)
    with pytest.raises(ValueError, match="exceeds the signal length"):
        theta_vec(np.zeros(10), np.zeros(10), 11)
