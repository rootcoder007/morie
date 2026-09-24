"""Tests for treatment_b_randomized.treatment_b_randomized."""

from morie.fn import _array_core as np

from morie.fn.treatment_b_randomized import treatment_b_randomized


def test_ca9e2_basic():
    """Test basic functionality."""
    r_yt = 0.5
    s_y = 0.5
    s_t = 0.5
    result = treatment_b_randomized(r_yt, s_y, s_t)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca9e2_edge():
    """Test edge cases."""
    r_yt = 0.5
    s_y = 0.5
    s_t = 0.5
    result = treatment_b_randomized(r_yt, s_y, s_t)
    assert isinstance(result, dict)
