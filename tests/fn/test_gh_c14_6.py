"""Tests for gh_c14_6.ghosal_ssp_post."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_6 import ghosal_ssp_post


def test_gh_c14_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 2.0
    result = ghosal_ssp_post(x, alpha=alpha)
    assert "estimate" in result
    assert "seen_weights" in result
    assert "total" in result
    assert "method" in result

    n = float(sum(x))
    expected_seen = [float(v) / (alpha + n) for v in x]
    expected_new = alpha / (alpha + n)

    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.allclose(np.asarray(result["seen_weights"], dtype=float), np.asarray(expected_seen, dtype=float))
    assert np.isclose(float(result["estimate"]), expected_new)
    assert np.isclose(float(result["total"]), sum(expected_seen) + expected_new)


def test_gh_c14_6_edge():
    """Test edge case with a single observation."""
    alpha = 2.0
    x = np.array([42.0])
    result = ghosal_ssp_post(x, alpha=alpha)

    # The function does not expose a key "n"; verify documented keys instead.
    assert "estimate" in result
    assert "seen_weights" in result
    assert "total" in result
    assert "method" in result

    n = 42.0
    expected_seen = [42.0 / (alpha + n)]
    expected_new = alpha / (alpha + n)

    assert np.allclose(np.asarray(result["seen_weights"], dtype=float), np.asarray(expected_seen, dtype=float))
    assert np.isclose(float(result["estimate"]), expected_new)
    assert np.isclose(float(result["total"]), sum(expected_seen) + expected_new)
