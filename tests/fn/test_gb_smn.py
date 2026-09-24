"""Tests for gb_smn.gibbons_smirnov_2sided."""

from morie.fn import _array_core as np

from morie.fn.gb_smn import gibbons_smirnov_2sided


def test_gb_smn_basic():
    """Test basic functionality."""
    d = 0.1
    m = 5
    n = 5
    result = gibbons_smirnov_2sided(d, m, n)
    assert isinstance(result, dict)
    assert "sf" in result or "sf" in result


def test_gb_smn_edge():
    """Test edge cases."""
    d = 0.1
    m = 5
    n = 5
    result = gibbons_smirnov_2sided(d, m, n)
    assert isinstance(result, dict)
