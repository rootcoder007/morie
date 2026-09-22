"""Tests for gh_c8_12.ghosal_crt_lower."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_12 import ghosal_crt_lower


def test_gh_c8_12_basic():
    """Test basic functionality across several (smoothness, n) pairs."""
    smoothness_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n_vals = np.array([10.0, 100.0, 1000.0, 10000.0, 100000.0])
    for s, n in zip(smoothness_vals, n_vals):
        result = ghosal_crt_lower(s, n)
        assert "estimate" in result
        est = float(np.asarray(result["estimate"], dtype=float))
        assert np.isfinite(est)
        # Independent recomputation of the literature formula
        expected_eps = float(n) ** (-float(s) / (2.0 * float(s) + 1.0))
        assert abs(est - expected_eps) < 1e-12 * max(1.0, abs(expected_eps))
        # Other documented keys
        assert "balance_gap" in result
        assert "exponent" in result
        assert "method" in result
        assert abs(float(result["exponent"]) - float(s) / (2.0 * float(s) + 1.0)) < 1e-12


def test_gh_c8_12_edge():
    """Test edge cases with scalar inputs."""
    result = ghosal_crt_lower(2.0, np.array([42.0]))
    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    expected_eps = 42.0 ** (-2.0 / (2.0 * 2.0 + 1.0))
    assert abs(est - expected_eps) < 1e-12 * max(1.0, abs(expected_eps))
