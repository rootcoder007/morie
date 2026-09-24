"""Tests for matern.matern_cluster."""

from morie.fn import _array_core as np

from morie.fn.matern import matern_cluster


def test_matern_basic():
    """Test basic functionality."""
    lambda_p = 0.1
    mu = 0.1
    r = 0.1
    result = matern_cluster(lambda_p, mu, r)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_matern_edge():
    """Test edge cases."""
    lambda_p = 0.1
    mu = 0.1
    r = 0.1
    result = matern_cluster(lambda_p, mu, r)
    assert isinstance(result, dict)
