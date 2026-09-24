"""Tests for alftrm.alphafold_triangle_mult."""

from morie.fn import _array_core as np

import math
import pytest

from morie.fn.alftrm import alphafold_triangle_mult


def test_alftrm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(0)
    n = 3
    cz = 2
    c = 2
    z = rng.normal(0, 1, (n, n, cz))
    wag = rng.normal(0, 1, (c, cz))
    wav = rng.normal(0, 1, (c, cz))
    wbg = rng.normal(0, 1, (c, cz))
    wbv = rng.normal(0, 1, (c, cz))
    wg = rng.normal(0, 1, (cz, cz))
    wo = rng.normal(0, 1, (cz, c))
    result = alphafold_triangle_mult(z, wag, wav, wbg, wbv, wg, wo)
    assert isinstance(result, dict)
    assert "z" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == n
    # Check shape of returned z: n x n x cz
    assert len(result["z"]) == n
    assert len(result["z"][0]) == n
    assert len(result["z"][0][0]) == cz
    # Check estimate is finite
    assert math.isfinite(result["estimate"])


def test_alftrm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(1)
    n = 3
    cz = 2
    c = 2
    z = rng.normal(0, 1, (n, n, cz))
    wag = rng.normal(0, 1, (c, cz))
    wav = rng.normal(0, 1, (c, cz))
    wbg = rng.normal(0, 1, (c, cz))
    wbv = rng.normal(0, 1, (c, cz))
    wg = rng.normal(0, 1, (cz, cz))
    wo = rng.normal(0, 1, (cz, c))
    # Invalid mode should raise ValueError
    with pytest.raises(ValueError):
        alphafold_triangle_mult(z, wag, wav, wbg, wbv, wg, wo, mode="invalid")
