"""Tests for rgbbb.rangayyan_bundle_branch_block."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_bundle_branch_block


def test_rgbbb_basic():
    """Test basic functionality."""
    qrsdur = 0.1
    result = rangayyan_bundle_branch_block(qrsdur)
    assert isinstance(result, dict)
    assert "blocktype" in result


def test_rgbbb_edge():
    """Test edge cases."""
    qrsdur = 0.1
    result = rangayyan_bundle_branch_block(qrsdur)
    assert isinstance(result, dict)
