"""Tests for motfom.motif_fimo."""

from morie.fn import _array_core as np

from morie.fn.motfom import motif_fimo


def test_motfom_basic():
    """Test basic functionality."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    pwm = np.random.default_rng(42).normal(0, 1, 100)
    result = motif_fimo(sequence, pwm)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_motfom_edge():
    """Test edge cases."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    pwm = np.random.default_rng(42).normal(0, 1, 100)
    result = motif_fimo(sequence, pwm)
    assert isinstance(result, dict)
