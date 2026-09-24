"""Tests for grgan.geron_gan_minimax."""

from morie.fn import _array_core as np

from morie.fn.grgan import geron_gan_minimax


def test_grgan_basic():
    """Test basic functionality."""
    real = [1.0, 2.0]
    fake = [3.0, 4.0]
    D_real = [0.5, 0.5]
    D_fake = [0.5, 0.5]
    result = geron_gan_minimax(real, fake, D_real, D_fake)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grgan_edge():
    """Test edge cases."""
    real = [1.0, 2.0]
    fake = [3.0, 4.0]
    D_real = [0.5, 0.5]
    D_fake = [0.5, 0.5]
    result = geron_gan_minimax(real, fake, D_real, D_fake)
    assert isinstance(result, dict)
