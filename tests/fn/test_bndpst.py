"""Tests for bndpst.bound_post_test."""

import pytest

from morie.fn import _array_core as np

from morie.fn.bndpst import bound_post_test


def test_bndpst_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    lower = rng.normal(0, 1, 100)
    upper = rng.normal(0, 1, 100)
    spec_test = rng.normal(0, 1, 30)
    with pytest.raises(NotImplementedError):
        bound_post_test(lower, upper, spec_test)


def test_bndpst_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    lower = rng.normal(0, 1, 100)
    upper = rng.normal(0, 1, 100)
    spec_test = rng.normal(0, 1, 30)
    with pytest.raises(NotImplementedError):
        bound_post_test(lower, upper, spec_test)
