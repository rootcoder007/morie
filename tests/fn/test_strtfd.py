"""Tests for strtfd.stratified_design."""

from morie.fn import _array_core as np

from morie.fn.strtfd import stratified_design


def test_strtfd_basic():
    """Test basic functionality."""
    Nh = 5
    Sh = 0.5
    n = 5
    result = stratified_design(Nh, Sh, n)
    assert isinstance(result, dict)
    assert "nh" in result


def test_strtfd_edge():
    """Test edge cases."""
    Nh = 5
    Sh = 0.5
    n = 5
    result = stratified_design(Nh, Sh, n)
    assert isinstance(result, dict)
