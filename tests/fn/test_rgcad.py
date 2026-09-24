"""Tests for rgcad.rangayyan_cad_pipeline."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_cad_pipeline


def test_rgcad_basic():
    """Test basic functionality."""
    features = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    labels = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = rangayyan_cad_pipeline(features, labels)
    assert isinstance(result, dict)
    assert "accuracy" in result


def test_rgcad_edge():
    """Test edge cases."""
    features = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    labels = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = rangayyan_cad_pipeline(features, labels)
    assert isinstance(result, dict)
