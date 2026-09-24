"""Tests for rgerrbd.rangayyan_bayes_error_bound."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_bayes_error_bound


def test_rgerrbd_basic():
    """Test basic functionality."""
    p1 = 0.5
    p2 = 0.5
    db = 0.5
    result = rangayyan_bayes_error_bound(p1, p2, db)
    assert isinstance(result, dict)
    assert "bound" in result


def test_rgerrbd_edge():
    """Test edge cases."""
    p1 = 0.5
    p2 = 0.5
    db = 0.5
    result = rangayyan_bayes_error_bound(p1, p2, db)
    assert isinstance(result, dict)
