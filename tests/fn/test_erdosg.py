"""Tests for erdosg.erdos_renyi_gnp."""

from morie.fn.erdosg import erdos_renyi_gnp


def test_erdosg_basic():
    """Test basic functionality."""
    n = 100
    p = 0.05
    result = erdos_renyi_gnp(n, p)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    m = n * (n - 1) // 2
    assert result["n"] == n
    assert result["expected_edges"] == m * p
    assert result["expected_degree"] == (n - 1) * p
    assert result["giant_threshold"] == 1.0 / n
    assert result["connectivity_threshold"] == 0.0 + (0 if n <= 1 else __import__("math").log(n) / n)
    assert result["method"] == "Erdos-Renyi G(n,p)"
    assert 0.0 <= result["density"] <= 1.0
    assert 0 <= result["edges"] <= m
    assert result["largest_component"] <= n


def test_erdosg_edge():
    """Test edge cases."""
    n = 100
    p = 0.0
    result = erdos_renyi_gnp(n, p)
    assert isinstance(result, dict)
    assert result["edges"] == 0
    assert result["density"] == 0.0
    assert result["n_components"] == n
    assert result["largest_component"] == 1
