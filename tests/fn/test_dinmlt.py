"""Tests for dinmlt.dino_multicrop."""

import math

from morie.fn import _array_core as np

from morie.fn.dinmlt import dino_multicrop


def test_dinmlt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    image = rng.normal(0, 1, (10, 4))  # V=10 (2 global + 8 local), d=4
    result = dino_multicrop(image, global_size=2, local_size=8)
    assert isinstance(result, dict)
    for key in ("estimate", "loss", "n_pairs", "teacher",
                "student_entropy", "V", "d"):
        assert key in result
    assert result["V"] == 10
    assert result["d"] == 4
    assert result["n_pairs"] == 18  # G * (V - 1) = 2 * 9
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["estimate"])


def test_dinmlt_edge():
    """Test edge case with an explicit centre vector."""
    rng = np.random.default_rng(42)
    image = rng.normal(0, 1, (3, 2))  # V=3 (1 global + 2 local), d=2
    center = rng.normal(0, 1, 2)
    result = dino_multicrop(image, global_size=1, local_size=2, center=center)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["V"] == 3
    assert result["d"] == 2
    assert result["n_pairs"] == 2  # G * (V - 1) = 1 * 2
    assert math.isfinite(result["loss"])
