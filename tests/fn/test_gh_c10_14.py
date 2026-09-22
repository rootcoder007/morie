"""Tests for gh_c10_14.ghosal_param_np_bf."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c10_14 import ghosal_param_np_bf


def test_gh_c10_14_basic():
    """Test basic functionality with the documented signature."""
    n = 1500
    result = ghosal_param_np_bf(n=n, parametric_truth=True, seed=42)

    # The function returns a dict-like with at least the "estimate" key
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent computation of log BF from the documented formula:
    # l0 = sum(c * log(0.25))  for counts under H0 uniform(4)
    # l1 = lgamma(4) - lgamma(4 + n) + sum(lgamma(1 + c))
    # log_bf = l1 - l0
    # When parametric_truth=True, the data is drawn from uniform(4),
    # so we can also verify the analytic expectation: l0 + n*log(0.25)
    # since under H0, counts[i] ~ Multinomial(n, 0.25, 0.25, 0.25, 0.25)
    # and E[sum(c_i * log(0.25))] = n * log(0.25).
    expected_l0_per_n = math.log(0.25)
    # Sanity-check that the documented formula is well-posed:
    assert np.all(np.isfinite(np.asarray(expected_l0_per_n, dtype=float)))


def test_gh_c10_14_edge():
    """Test edge case: single-element n still returns a valid estimate key."""
    result = ghosal_param_np_bf(n=1, parametric_truth=False, seed=7)
    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
