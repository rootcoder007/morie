"""Tests for gh_gp_orn_uhl.ghosal_gp_ornstein_uhlenbeck."""

from morie.fn import _array_core as np

from morie.fn.gh_gp_orn_uhl import ghosal_gp_ornstein_uhlenbeck


def test_gh_gp_orn_uhl_basic():
    """Test basic functionality."""
    theta = 1.0
    ts = (0.2, 0.5, 0.9)
    result = ghosal_gp_ornstein_uhlenbeck(theta=theta, ts=ts)
    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
    # Independent computation per the docstring: K(s,t) = exp(-theta|s-t|) / (2*theta)
    s, t = 0.2, 0.5
    expected = np.exp(-theta * abs(s - t)) / (2.0 * theta)
    assert float(np.asarray(result["estimate"]).reshape(-1)[0]) == pytest_approx(expected)


def test_gh_gp_orn_uhl_edge():
    """Test edge cases (single, equal-ish triple)."""
    theta = 1.0
    ts = (0.2, 0.2, 0.2)
    result = ghosal_gp_ornstein_uhlenbeck(theta=theta, ts=ts)
    assert "estimate" in result
    # Independent computation per the docstring formula.
    s, t, u = ts
    expected = np.exp(-theta * abs(s - t)) / (2.0 * theta)
    assert float(np.asarray(result["estimate"]).reshape(-1)[0]) == pytest_approx(expected)


def pytest_approx(x):
    """Tiny local helper since pytest is not imported here."""
    import math
    return x
