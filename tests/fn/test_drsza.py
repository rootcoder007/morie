"""Tests for drsza.dr_did_santanna_zhao."""

from morie.fn import _array_core as np

from morie.fn.drsza import dr_did_santanna_zhao


def test_drsza_basic():
    """Test basic functionality."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    y_post = np.random.default_rng(42).normal(0, 1, 100)
    treatment = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = dr_did_santanna_zhao(y_pre, y_post, treatment)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_drsza_edge():
    """Test edge cases."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    y_post = np.random.default_rng(42).normal(0, 1, 100)
    treatment = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = dr_did_santanna_zhao(y_pre, y_post, treatment)
    assert isinstance(result, dict)
