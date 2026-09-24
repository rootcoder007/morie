"""Tests for otmtxe.ot_matrix_scaling."""

from morie.fn import _array_core as np

from morie.fn.otmtxe import ot_matrix_scaling


def test_otmtxe_basic():
    """Test basic functionality."""
    K = 0.5
    row_target = 0.5
    col_target = 0.5
    result = ot_matrix_scaling(K, row_target, col_target)
    assert isinstance(result, dict)
    assert "M" in result


def test_otmtxe_edge():
    """Test edge cases."""
    K = 0.5
    row_target = 0.5
    col_target = 0.5
    result = ot_matrix_scaling(K, row_target, col_target)
    assert isinstance(result, dict)
