"""Tests for km090.kamath_ch6_co_occurrence_bias."""

import math
import pytest

from morie.fn import _array_core as np

from morie.fn.km090 import kamath_ch6_co_occurrence_bias


def test_km090_basic():
    """Test basic functionality."""
    w = "nurse"
    A_i = ["nurse doctor", "nurse teacher", "doctor lawyer"]
    A_j = ["nurse engineer", "lawyer nurse", "scientist doctor"]
    result = kamath_ch6_co_occurrence_bias(w, A_i, A_j)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "p_given_Ai" in result
    assert "p_given_Aj" in result
    assert "count_Ai" in result
    assert "count_Aj" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["p_given_Ai"] <= 1.0
    assert 0.0 <= result["p_given_Aj"] <= 1.0
    assert result["n"] > 0


def test_km090_edge():
    """Test edge cases."""
    w = "absent"
    A_i = ["nurse doctor", "teacher lawyer"]
    A_j = ["nurse doctor", "teacher lawyer"]
    with pytest.raises(Exception):
        kamath_ch6_co_occurrence_bias(w, A_i, A_j)
