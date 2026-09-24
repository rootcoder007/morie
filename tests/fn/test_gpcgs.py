"""Tests for gpcgs.gp_classification_svgp."""

import math

from morie.fn import _array_core as np

from morie.fn.gpcgs import gp_classification_svgp


def test_gpcgs_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = [int(v) for v in rng.integers(0, 2, n)]
    X_test = rng.normal(0, 1, (10, p))
    result = gp_classification_svgp(X, y, X_test, 3)
    assert isinstance(result, dict)
    assert 'elbo' in result
    assert math.isfinite(result['elbo'])
    assert 'kl' in result
    assert math.isfinite(result['kl'])


def test_gpcgs_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 20, 2
    X = rng.normal(0, 1, (n, p))
    y = [int(v) for v in rng.integers(0, 2, n)]
    result = gp_classification_svgp(X, y, None, 2)
    assert isinstance(result, dict)
    assert 'elbo' in result
    assert math.isfinite(result['elbo'])
    assert 'kl' in result
    assert math.isfinite(result['kl'])
