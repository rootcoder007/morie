"""Tests for _bits._bits."""

import math

from morie.fn import _array_core as np
from morie.fn._bits import _bits


def test_ghs026_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m = 3
    n_cells = 2 ** m
    raw = rng.uniform(0, 1, n_cells)
    total = np.sum(raw)
    masses = [raw[i] / total for i in range(n_cells)]
    x = 0.3
    result = _bits(x, masses, m)
    assert isinstance(result, dict)
    expected_keys = {"estimate", "distribution", "cell_index", "total_mass", "method"}
    assert expected_keys.issubset(set(result.keys()))
    assert math.isfinite(result["estimate"])
    assert 0 <= result["cell_index"] < n_cells
    assert math.isfinite(result["total_mass"])


def test_ghs026_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    m = 0
    masses = [1.0]
    x = 0.0
    result = _bits(x, masses, m)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["cell_index"] == 0
