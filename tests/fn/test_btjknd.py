"""Tests for btjknd.boot_jackknife_d."""

import math

from morie.fn import _array_core as np

from morie.fn.btjknd import boot_jackknife_d


def test_btjknd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    d = 2
    nsub = math.comb(n, d)
    theta = rng.normal(0, 1, nsub)
    result = boot_jackknife_d(theta, n, d)
    assert isinstance(result, dict)
    numeric_vals = [v for v in result.values() if isinstance(v, (int, float))]
    assert len(numeric_vals) > 0
    assert all(math.isfinite(v) for v in numeric_vals)


def test_btjknd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 20
    d = 2
    nsub = math.comb(n, d)
    theta = rng.normal(0, 1, nsub)
    result = boot_jackknife_d(theta, n, d)
    assert isinstance(result, dict)
    numeric_vals = [v for v in result.values() if isinstance(v, (int, float))]
    assert len(numeric_vals) > 0
    assert all(math.isfinite(v) for v in numeric_vals)
