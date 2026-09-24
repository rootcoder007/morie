"""Tests for rgnmfch.rangayyan_nmf_channel_sel."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_nmf_channel_sel


def test_rgnmfch_basic():
    """Test basic functionality."""
    trials = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    nselect = 5
    result = rangayyan_nmf_channel_sel(trials, nselect)
    assert isinstance(result, dict)
    assert "selected" in result


def test_rgnmfch_edge():
    """Test edge cases."""
    trials = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    nselect = 5
    result = rangayyan_nmf_channel_sel(trials, nselect)
    assert isinstance(result, dict)
