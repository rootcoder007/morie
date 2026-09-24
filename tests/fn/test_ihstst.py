"""Tests for ihstst.ihs_test."""

import math

from morie.fn import _array_core as np
from morie.fn.ihstst import ihs_test


def test_ihstst_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, L = 40, 50
    hap = rng.integers(0, 2, (n, L))
    core = L // 2
    result = ihs_test(hap, core)
    assert "estimate" in result
    assert "ihs_unstandardized" in result
    assert "ihh_a" in result
    assert "ihh_d" in result
    assert "daf" in result
    assert "truncated_a" in result
    assert "truncated_d" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["ihs_unstandardized"])
    assert result["ihh_a"] > 0
    assert result["ihh_d"] > 0
    assert 0.0 <= result["daf"] <= 1.0
    assert result["standardized"] is False
    assert result["core"] == core


def test_ihstst_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, L = 40, 50
    hap = rng.integers(0, 2, (n, L))
    core = L // 2
    result = ihs_test(hap, core, standardize=(0.0, 1.0))
    assert result["standardized"] is True
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["ihs_unstandardized"])
    assert "ihh_a" in result and "ihh_d" in result
    assert result["ihh_a"] > 0 and result["ihh_d"] > 0
