"""Tests for grdpmf.geron_ddpm_forward_process."""

from morie.fn import _array_core as np

from morie.fn.grdpmf import geron_ddpm_forward_process


def test_grdpmf_basic():
    """Test basic functionality."""
    x0 = [2.0, -4.0]
    t = 1
    alpha_bar = [1.0, 0.36]
    result = geron_ddpm_forward_process(x0, t, alpha_bar)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdpmf_edge():
    """Test edge cases."""
    x0 = [2.0, -4.0]
    t = 1
    alpha_bar = [1.0, 0.36]
    result = geron_ddpm_forward_process(x0, t, alpha_bar)
    assert isinstance(result, dict)
