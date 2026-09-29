"""Tests for alfrcl.alphafold_recycle_loss."""

from morie.fn import _array_core as np
from morie.fn.alfrcl import alphafold_recycle_loss


def test_alfrcl_basic():
    """Test basic functionality."""
    losses = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_recycle_loss(losses)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_alfrcl_edge():
    """Test edge cases."""
    losses = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_recycle_loss(losses)
    assert isinstance(result, dict)


def test_alfrcl_average_and_unbiased_single_iteration():
    import pytest

    losses = [2.0, 1.4, 0.9, 0.7]
    r = alphafold_recycle_loss(losses)
    assert r["estimate"] == pytest.approx(sum(losses) / 4, rel=1e-15)
    singles = [alphafold_recycle_loss(losses, nprime=k)["estimate"] for k in range(1, 5)]
    assert singles == losses
    assert sum(singles) / 4 == pytest.approx(r["average"], rel=1e-15)
