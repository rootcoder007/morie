"""Tests for tmscore.tm_score."""

from morie.fn import _array_core as np

from morie.fn.tmscore import tm_score


def test_tmscore_basic():
    """Test basic functionality."""
    coords1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    coords2 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tm_score(coords1, coords2)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmscore_edge():
    """Test edge cases."""
    coords1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    coords2 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tm_score(coords1, coords2)
    assert isinstance(result, dict)
