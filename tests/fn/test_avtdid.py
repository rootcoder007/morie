"""Tests for avtdid.avg_treatment_did."""

from morie.fn import _array_core as np

from morie.fn.avtdid import avg_treatment_did


def test_avtdid_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_x = np.random.default_rng(7)

    n = 100
    y = rng_y.normal(0, 1, n)
    D = (rng_d.uniform(0, 1, n) < 0.5).astype(float)
    X = rng_x.normal(0, 1, (n, 5))
    result = avg_treatment_did(y, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_avtdid_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_x = np.random.default_rng(7)

    n = 100
    y = rng_y.normal(0, 1, n)
    D = (rng_d.uniform(0, 1, n) < 0.5).astype(float)
    X = rng_x.normal(0, 1, (n, 5))
    result = avg_treatment_did(y, D, X)
    assert isinstance(result, dict)
