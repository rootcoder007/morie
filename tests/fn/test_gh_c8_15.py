"""Tests for gh_c8_15.ghosal_alpha_pst_crt."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c8_15 import ghosal_alpha_pst_crt


def test_gh_c8_15_basic():
    """Test basic functionality with default arguments."""
    result = ghosal_alpha_pst_crt()
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert math.isfinite(estimate)
    # The rate of alpha-posterior contraction should be near 1
    # (same n^{-1} rate as the full posterior for alpha < 1).
    assert abs(estimate - 1.0) < 0.15
    # The function records a variance-per-n sequence.
    assert "var_by_n" in result
    var_seq = [float(v) for v in result["var_by_n"]]
    assert len(var_seq) == 2
    # Independently recompute the rate from the recorded var_by_n
    # using the documented log-ratio formula and verify it matches.
    ns = (100, 10000)
    expected_rate = math.log(var_seq[0] / var_seq[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    assert abs(expected_rate - estimate) < 1e-12


def test_gh_c8_15_edge():
    """Test edge cases using explicit keyword arguments."""
    result = ghosal_alpha_pst_crt(theta0=0.3, alpha=0.5, ns=(50, 500), seed=1)
    assert "estimate" in result
    assert "method" in result
    assert isinstance(result["method"], str)
    assert result["method"].startswith("alpha-posterior")
