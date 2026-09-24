"""Tests for km114.kamath_ch8_bleu_precision."""

from morie.fn import _array_core as np

from morie.fn.km114 import kamath_ch8_bleu_precision


def test_km114_basic():
    """Test basic functionality with two n-gram orders."""
    n_grams = [[3, 4], [1, 3]]
    result = kamath_ch8_bleu_precision(n_grams)
    assert isinstance(result, dict)
    assert "p_n" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 2
    assert len(result["p_n"]) == 2
    # 3/4
    assert abs(result["p_n"][0] - 0.75) < 1e-12
    # 1/3
    assert abs(result["p_n"][1] - 1 / 3) < 1e-12


def test_km114_edge():
    """Test edge case with a single n-gram order."""
    n_grams = [[2, 5]]
    result = kamath_ch8_bleu_precision(n_grams)
    assert isinstance(result, dict)
    assert result["n"] == 1
    assert len(result["p_n"]) == 1
    # 2/5
    assert abs(result["p_n"][0] - 0.4) < 1e-12
