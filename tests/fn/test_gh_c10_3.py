"""Tests for gh_c10_3.ghosal_param_rate."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_3 import ghosal_param_rate


def test_gh_c10_3_basic():
    """Test basic functionality."""
    result = ghosal_param_rate(d_true=2, ns=(100, 1000, 10000),
                               lam=1.0, seed=42)
    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
    # The literature rate is 1.0 (parametric sqrt(d/n) rate);
    # the estimator should recover it within tolerance.
    assert abs(float(est) - 1.0) < 0.25
    # Additional documented keys must be present.
    assert "risk_by_n" in result
    assert "parametric" in result
    assert "method" in result
    assert np.asarray(result["risk_by_n"]).shape == (3,)


def test_gh_c10_3_edge():
    """Test edge cases with a single small n."""
    result = ghosal_param_rate(d_true=1, ns=(10, 100), lam=1.0, seed=0)
    assert "risk_by_n" in result
    assert len(result["risk_by_n"]) == 2
    risks = [float(r) for r in result["risk_by_n"]]
    assert all(np.isfinite(r) and r >= 0.0 for r in risks)
