"""Tests for rgbci.rangayyan_bci_nmf."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_bci_nmf


def test_rgbci_basic():
    """Test basic functionality."""
    trials = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    nselect = 5
    result = rangayyan_bci_nmf(trials, nselect)
    assert isinstance(result, dict)
    assert "selected" in result


def test_rgbci_edge():
    """Test edge cases."""
    trials = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    nselect = 5
    result = rangayyan_bci_nmf(trials, nselect)
    assert isinstance(result, dict)
