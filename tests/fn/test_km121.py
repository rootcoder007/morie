"""Tests for km121.kamath_ch8_bertscore_f1."""

import math

from morie.fn import _array_core as np

from morie.fn.km121 import kamath_ch8_bertscore_f1


def test_km121_basic():
    """Test basic functionality."""
    P_BERT = 0.8
    R_BERT = 0.6
    result = kamath_ch8_bertscore_f1(P_BERT, R_BERT)
    assert isinstance(result, dict)
    assert "estimate" in result
    expected = 2.0 * 0.8 * 0.6 / (0.8 + 0.6)
    assert math.isclose(result["estimate"], expected, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isfinite(result["estimate"])


def test_km121_edge():
    """Test edge cases with valid boundary values."""
    # Very small but strictly positive precision and recall
    P_BERT = 1e-12
    R_BERT = 1e-12
    result = kamath_ch8_bertscore_f1(P_BERT, R_BERT)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "precision" in result
    assert "recall" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
