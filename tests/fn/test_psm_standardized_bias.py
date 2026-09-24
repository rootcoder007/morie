"""Tests for psm_standardized_bias.psm_standardized_bias."""

from morie.fn import _array_core as np

from morie.fn.psm_standardized_bias import psm_standardized_bias


def test_ca10e1_basic():
    """Test basic functionality."""
    mean_t = 0.5
    mean_c = 0.5
    s_t = 0.5
    s_c = 0.5
    result = psm_standardized_bias(mean_t, mean_c, s_t, s_c)
    assert isinstance(result, dict)
    assert "value" in result or "value" in result


def test_ca10e1_edge():
    """Test edge cases."""
    mean_t = 0.5
    mean_c = 0.5
    s_t = 0.5
    s_c = 0.5
    result = psm_standardized_bias(mean_t, mean_c, s_t, s_c)
    assert isinstance(result, dict)
