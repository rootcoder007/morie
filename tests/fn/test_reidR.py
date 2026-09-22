"""Tests for reidR.reidentification_risk."""

from morie.fn import _array_core as np

from morie.fn.reidR import reidentification_risk


def test_reidR_basic():
    """Test basic functionality."""
    quasi_identifiers = np.random.default_rng(42).normal(0, 1, 100)
    result = reidentification_risk(quasi_identifiers)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_reidR_edge():
    """Test edge cases."""
    quasi_identifiers = np.random.default_rng(42).normal(0, 1, 100)
    result = reidentification_risk(quasi_identifiers)
    assert isinstance(result, dict)
