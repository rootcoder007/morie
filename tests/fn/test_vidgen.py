"""Tests for vidgen.video_diffusion."""

from morie.fn import _array_core as np

from morie.fn.vidgen import video_diffusion


def test_vidgen_basic():
    """Test basic functionality."""
    x_hat = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    observed = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    index = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = video_diffusion(x_hat, observed, index)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vidgen_edge():
    """Test edge cases."""
    x_hat = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    observed = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    index = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = video_diffusion(x_hat, observed, index)
    assert isinstance(result, dict)
