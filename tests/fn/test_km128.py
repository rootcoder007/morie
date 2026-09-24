"""Tests for km128.kamath_ch8_pass_at_k."""

from morie.fn.km128 import kamath_ch8_pass_at_k


def test_km128_basic():
    """Test basic functionality."""
    result = kamath_ch8_pass_at_k(100, 50, 5)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "fail_probability" in result
    assert "n_samples" in result
    assert "n_correct" in result
    assert "k" in result
    assert "n" in result
    assert "method" in result
    assert 0.0 <= result["estimate"] <= 1.0
    assert 0.0 <= result["fail_probability"] <= 1.0
    assert result["n_samples"] == 100
    assert result["n_correct"] == 50
    assert result["k"] == 5


def test_km128_edge():
    """Test edge cases."""
    # c = 0 means all samples failed, so pass@k should be exactly 0
    result = kamath_ch8_pass_at_k(100, 0, 5)
    assert isinstance(result, dict)
    assert result["estimate"] == 0.0
    assert result["fail_probability"] == 1.0

    # c = n means all samples passed, so pass@k should be exactly 1
    result2 = kamath_ch8_pass_at_k(100, 100, 5)
    assert isinstance(result2, dict)
    assert result2["estimate"] == 1.0
    assert result2["fail_probability"] == 0.0
