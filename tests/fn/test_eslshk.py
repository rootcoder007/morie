"""Tests for eslshk.esl_shrinkage."""

from morie.fn import _array_core as np

from morie.fn.eslshk import esl_shrinkage


def test_eslshk_basic():
    """Test basic functionality."""
    nu = 0.1
    M = 100
    result = esl_shrinkage(nu, M=M)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "capacity" in result
    assert "nu" in result
    assert "M" in result
    assert "required_M" in result
    assert "regime" in result
    assert "method" in result
    # independent arithmetic: nu * M computed inline
    assert result["capacity"] == nu * M
    assert result["nu"] == nu
    assert result["M"] == M
    assert result["estimate"] == nu * M
    assert result["regime"] == "shrunk (nu <= 0.1): prefer many stages"


def test_eslshk_edge():
    """Test edge cases."""
    # target_capacity branch: nu=0.05, target=10 -> required_M = ceil(10/0.05) = 200
    nu = 0.05
    target_capacity = 10.0
    result = esl_shrinkage(nu, target_capacity=target_capacity)
    assert isinstance(result, dict)
    # independent computation of expected required_M
    expected_required_M = int(np.ceil(target_capacity / nu))
    assert result["required_M"] == expected_required_M
    assert result["required_M"] == 200
    assert result["estimate"] == expected_required_M
    # capacity is None because M was not given
    assert result["capacity"] is None
    assert result["M"] is None
    assert result["nu"] == nu
