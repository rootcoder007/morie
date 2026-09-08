"""Tests for regms.regime_switching."""

from morie.fn import _array_core as np
import pytest

from morie.fn.regms import regime_switching


def _dgp(seed, n=600, mus=(-1.0, 2.0), sigmas=(0.5, 0.5), p_stay=0.95):
    rng = np.random.default_rng(seed)
    state, y = 0, np.empty(n)
    for t in range(n):
        if rng.uniform() > p_stay:
            state = 1 - state
        y[t] = mus[state] + sigmas[state] * rng.standard_normal()
    return y


def test_regms_recovers_the_two_means():
    """Hamilton (1989) regime means, sorted to kill label switching."""
    r = regime_switching(_dgp(1), k_regimes=2)
    mu = np.sort(np.asarray(r["mu"], dtype=float).ravel())
    assert mu[0] == pytest.approx(-1.0, abs=0.2)
    assert mu[1] == pytest.approx(2.0, abs=0.2)


def test_regms_transition_matrix_is_sticky_and_stochastic():
    r = regime_switching(_dgp(2), k_regimes=2)
    P = np.asarray(r["transition"], dtype=float)
    assert P.shape == (2, 2)
    # The module documents the row-stochastic convention
    # P[i, j] = P(next = j | current = i).
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    # p_stay = 0.95 in the DGP: both diagonal entries must be large.
    assert min(P[0, 0], P[1, 1]) > 0.8


def test_regms_smoothed_probabilities_track_the_truth():
    y = _dgp(3)
    r = regime_switching(y, k_regimes=2)
    probs = np.asarray(r["smoothed_probabilities"], dtype=float)
    # Classify by the smoothed mode and compare with the observable proxy
    # (y > 0.5 separates the two means almost perfectly at sigma = 0.5).
    hard = probs.argmax(axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int)
    truth = (y > 0.5).astype(int)
    agree = max(np.mean(hard == truth), np.mean(hard != truth))  # labels may be flipped
    assert agree > 0.9


def test_regms_rejects_short_series():
    with pytest.raises(ValueError, match="at least"):
        regime_switching(np.arange(6.0), k_regimes=2)
