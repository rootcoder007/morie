"""Tests for gh_c13_4.ghosal_bp_discrete."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_4 import ghosal_bp_discrete


def test_gh_c13_4_basic():
    """Test basic functionality."""
    hazards0 = (0.1, 0.2, 0.3)
    result = ghosal_bp_discrete(hazards0=hazards0, c=4.0, seed=42)
    assert "estimate" in result
    assert "mean_by_time" in result
    assert "prior_mean_gap" in result
    assert "method" in result
    est = float(result["estimate"])
    expected_est = float(sum(hazards0))
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))
    # estimate = sum of per-time Monte Carlo means; each per-time mean is an
    # MC estimate of h_k, so the estimate should be close to sum(h_k).
    # With n_sim=2000 and c=4.0 the MC error per component is small (~1/sqrt(8000)).
    assert abs(est - expected_est) < 0.1


def test_gh_c13_4_edge():
    """Test edge cases."""
    hazards0 = (0.5,)
    result = ghosal_bp_discrete(hazards0=hazards0, c=4.0, seed=42)
    assert "estimate" in result
    est = float(result["estimate"])
    # Single-time-series: estimate should be close to 0.5.
    assert abs(est - 0.5) < 0.1
    # mean_by_time should have length 1.
    mbt = result["mean_by_time"]
    assert len(mbt) == 1
