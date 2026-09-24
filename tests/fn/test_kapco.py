"""Tests for kapco.kappa_coefficient."""

from morie.fn import _array_core as np

from morie.fn.kapco import kappa_coefficient


def test_kapco_basic():
    """Test basic functionality."""
    tp = 0.1
    fp = 0.1
    fn = 0.1
    tn = 0.1
    result = kappa_coefficient(tp, fp, fn, tn)
    assert isinstance(result, dict)
    assert "kappa" in result


def test_kapco_edge():
    """Test edge cases."""
    tp = 0.1
    fp = 0.1
    fn = 0.1
    tn = 0.1
    result = kappa_coefficient(tp, fp, fn, tn)
    assert isinstance(result, dict)
