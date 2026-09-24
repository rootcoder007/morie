"""Tests for cndent.conditional_entropy."""

import math
import pytest

from morie.fn import _array_core as np

from morie.fn.cndent import conditional_entropy


def test_cndent_basic():
    """Test basic functionality with a valid 2-D joint pmf."""
    rng = np.random.default_rng(42)
    nx, ny = 3, 4
    raw = rng.uniform(0.0, 1.0, (nx, ny))
    pxy = [[float(raw[i][j]) for j in range(ny)] for i in range(nx)]

    result = conditional_entropy(pxy)

    for key in ("estimate", "hxy", "hx", "hy", "n", "method"):
        assert key in result

    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["hxy"])
    assert math.isfinite(result["hx"])
    assert math.isfinite(result["hy"])
    assert result["n"] == nx * ny
    assert abs(result["estimate"] - (result["hxy"] - result["hx"])) < 1e-12


def test_cndent_edge():
    """Test edge case: pmf with zero total mass raises ValueError."""
    pxy = [[0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError):
        conditional_entropy(pxy)
