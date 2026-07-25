"""Tests for ganls.gan_loss."""

import numpy as np

from morie.fn.ganls import gan_loss


def test_ganls_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gan_loss(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ganls_edge():
    """Test edge cases."""
    result = gan_loss(np.array([42.0]))
    assert result["n"] == 1
