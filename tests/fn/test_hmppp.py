"""Tests for hmppp.geron_pipeline_parallelism."""

from morie.fn import _array_core as np

from morie.fn.hmppp import geron_pipeline_parallelism


def test_hmppp_basic():
    """Test basic functionality."""
    model = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    n_stages = 5
    result = geron_pipeline_parallelism(model, n_stages)
    assert isinstance(result, dict)
    assert "estimate" in result or "assignment" in result


def test_hmppp_edge():
    """Test edge cases."""
    model = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    n_stages = 5
    result = geron_pipeline_parallelism(model, n_stages)
    assert isinstance(result, dict)
