"""Tests for grgmem.geron_gmm_em_step."""

from morie.fn import _array_core as np

from morie.fn.grgmem import geron_gmm_em_step


def test_grgmem_basic():
    """Test basic functionality."""
    X = [[0.0], [0.4], [5.0], [5.5]]
    pi = [0.5, 0.5]
    means = [[1.0], [4.0]]
    covars = [[[1.0]], [[1.0]]]
    result = geron_gmm_em_step(X, pi, means, covars)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grgmem_edge():
    """Test edge cases."""
    X = [[0.0], [0.4], [5.0], [5.5]]
    pi = [0.5, 0.5]
    means = [[1.0], [4.0]]
    covars = [[[1.0]], [[1.0]]]
    result = geron_gmm_em_step(X, pi, means, covars)
    assert isinstance(result, dict)
