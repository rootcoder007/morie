"""Tests for km049.kamath_ch3_top1_prompt_metric."""

import pytest

from morie.fn import _array_core as np

from morie.fn.km049 import kamath_ch3_top1_prompt_metric


def test_km049_basic():
    """Test basic functionality."""
    R = [("a", "pos"), ("b", "neg"), ("c", "pos"), ("d", "pos")]
    t = "T1"

    def P_LM(x, t):
        return {"pos": 0.7, "neg": 0.3}

    result = kamath_ch3_top1_prompt_metric(R, t, P_LM)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "n_correct" in result
    assert "n" in result
    assert result["n"] == 4
    assert result["n_correct"] == 3
    assert 0.0 <= result["estimate"] <= 1.0


def test_km049_edge():
    """Test edge cases."""
    with pytest.raises(ValueError):
        kamath_ch3_top1_prompt_metric(
            [], "T1", lambda x, t: {"pos": 1.0, "neg": 0.0})
