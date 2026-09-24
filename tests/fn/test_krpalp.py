"""Tests for krpalp.krippendorff_alpha."""

from morie.fn import _array_core as np

from morie.fn.krpalp import krippendorff_alpha


def test_krpalp_basic():
    """Test basic functionality."""
    data = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = krippendorff_alpha(data)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_krpalp_edge():
    """Test edge cases."""
    data = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = krippendorff_alpha(data)
    assert isinstance(result, dict)
