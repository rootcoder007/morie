"""Tests for grmcd.geron_mc_dropout."""

from morie.fn import _array_core as np

from morie.fn.grmcd import geron_mc_dropout


def test_grmcd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # 3-D logits expected by mcdrop / _core.mcdrop:
    # (n_samples, n_mc_runs, n_classes)
    data = rng.normal(0, 1, (40, 10, 5))
    result = geron_mc_dropout(data)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_grmcd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (10, 5, 3))
    result = geron_mc_dropout(data)
    assert isinstance(result, dict)
    assert len(result) > 0
