"""Tests for km095.kamath_ch6_gender_direction."""

import math

from morie.fn.km095 import kamath_ch6_gender_direction


def test_km095_basic():
    """Test basic functionality with a small gender pair set."""
    E = {
        "she": [0.0, 1.0],
        "he": [2.0, 1.0],
        "her": [0.0, 1.0],
        "his": [2.0, 1.0],
        "woman": [0.0, 1.0],
        "man": [2.0, 1.0],
    }
    A = [("she", "he"), ("her", "his"), ("woman", "man")]
    result = kamath_ch6_gender_direction(A, E)
    assert isinstance(result, dict)
    assert result["n"] == 3
    assert len(result["g"]) == 2
    assert result["g"] == [2.0, 0.0]
    assert math.isfinite(result["norm"])
    assert result["norm"] == 2.0
    assert len(result["per_pair"]) == 3
    assert result["per_pair"] == [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]
    assert result["estimate"] == 2.0
    assert result["degenerate"] is False
    assert result["method"] == "gender direction (Kamath Eq 6.19)"


def test_km095_edge():
    """Test edge case with a single pair matches the docstring example."""
    E = {"she": [0.0, 1.0], "he": [2.0, 1.0]}
    A = [("she", "he")]
    result = kamath_ch6_gender_direction(A, E)
    assert isinstance(result, dict)
    assert result["n"] == 1
    assert result["g"] == [2.0, 0.0]
    assert result["norm"] == 2.0
    assert result["per_pair"] == [[2.0, 0.0]]
    assert result["estimate"] == 2.0
    assert result["degenerate"] is False
