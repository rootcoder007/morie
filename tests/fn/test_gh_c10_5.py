"""Tests for gh_c10_5.ghosal_wn_adapt."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_5 import ghosal_wn_adapt


def test_gh_c10_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_wn_adapt(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c10_5_edge():
    """Test edge cases: single-observation input yields a single inclusion probability."""
    import math

    y = np.array([42.0])
    result = ghosal_wn_adapt(y)

    assert "inclusion_probs" in result
    incl = np.asarray(result["inclusion_probs"], dtype=float)
    assert incl.shape == (1,)

    # Independent computation of the single inclusion probability.
    yk = 42.0
    v = 1.0 / 400.0
    pi_incl = 0.2
    tau2 = 1.0
    l1 = -0.5 * math.log(2 * math.pi * (v + tau2)) \
         - 0.5 * yk * yk / (v + tau2) + math.log(pi_incl)
    l0 = -0.5 * math.log(2 * math.pi * v) \
         - 0.5 * yk * yk / v + math.log(1.0 - pi_incl)
    expected_prob = 1.0 / (1.0 + math.exp(l0 - l1))

    assert np.isclose(incl[0], expected_prob)
    assert np.isclose(np.asarray(result["estimate"], dtype=float), expected_prob)
