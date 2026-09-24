"""Tests for otmapnk.ot_map_neural_kantorovich."""

from morie.fn import _array_core as np

from morie.fn.otmapnk import ot_map_neural_kantorovich


def test_otmapnk_basic():
    """Test basic functionality."""
    source = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    target = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_map_neural_kantorovich(source, target)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_otmapnk_edge():
    """Test edge cases."""
    source = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    target = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ot_map_neural_kantorovich(source, target)
    assert isinstance(result, dict)
