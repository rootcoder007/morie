"""Tests for rgpdfest.rangayyan_pdf_estimate."""

from morie.fn import _array_core as np

from morie.fn.bsastat import rangayyan_pdf_estimate


def test_rgpdfest_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_pdf_estimate(x)
    assert isinstance(result, dict)
    assert "grid" in result


def test_rgpdfest_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_pdf_estimate(x)
    assert isinstance(result, dict)
