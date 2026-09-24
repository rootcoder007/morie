"""Tests for ovbnk.oster_omitted_bias_bound."""

from morie.fn import _array_core as np

from morie.fn.ovbnk import oster_omitted_bias_bound


def test_ovbnk_basic():
    """Test basic functionality."""
    beta_short = 0.5
    beta_long = 0.5
    R_short = 0.5
    R_long = 1
    result = oster_omitted_bias_bound(beta_short, beta_long, R_short, R_long)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ovbnk_edge():
    """Test edge cases."""
    beta_short = 0.5
    beta_long = 0.5
    R_short = 0.5
    R_long = 1
    result = oster_omitted_bias_bound(beta_short, beta_long, R_short, R_long)
    assert isinstance(result, dict)
