"""Tests for smwgrp.small_worldness."""

from morie.fn import _array_core as np

from morie.fn.smwgrp import small_worldness


def test_smwgrp_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = small_worldness(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_smwgrp_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = small_worldness(A)
    assert isinstance(result, dict)
