"""Tests for spkpe.schabenberger_kriging_pred_error."""

from morie.fn import _array_core as np

from morie.fn.spkpe import schabenberger_kriging_pred_error


def test_spkpe_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    target = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = schabenberger_kriging_pred_error(coords, z, target)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_spkpe_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    target = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = schabenberger_kriging_pred_error(coords, z, target)
    assert isinstance(result, dict)
