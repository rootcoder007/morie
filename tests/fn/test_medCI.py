"""Tests for medCI.asymmetric_indirect_ci."""

from morie.fn import _array_core as np

from morie.fn.medCI import asymmetric_indirect_ci


def test_medCI_basic():
    """Test basic functionality."""
    a = 0.5
    b = 0.5
    sa = 0.5
    sb = 0.5
    result = asymmetric_indirect_ci(a, b, sa, sb)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_medCI_edge():
    """Test edge cases."""
    a = 0.5
    b = 0.5
    sa = 0.5
    sb = 0.5
    result = asymmetric_indirect_ci(a, b, sa, sb)
    assert isinstance(result, dict)
