"""Tests for mafrt.ma_freeman_tukey."""

import math

from morie.fn import _array_core as np

from morie.fn.mafrt import ma_freeman_tukey


def test_mafrt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_studies = 20
    n = list(rng.integers(50, 200, n_studies))
    # x_i is an event count with 0 <= x_i <= n_i
    x = [int(rng.integers(0, int(ni) + 1, endpoint=True)) for ni in n]
    result = ma_freeman_tukey(x, n)
    assert isinstance(result, dict)
    for key in ("ft", "var", "se", "k"):
        assert key in result
    assert result["k"] == n_studies
    assert len(result["ft"]) == n_studies
    assert len(result["var"]) == n_studies
    assert len(result["se"]) == n_studies
    for v in result["ft"]:
        assert math.isfinite(v)
    for v in result["var"]:
        assert math.isfinite(v) and v > 0
    for v in result["se"]:
        assert math.isfinite(v) and v > 0


def test_mafrt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n_studies = 5
    n = list(rng.integers(50, 200, n_studies))
    x = [int(rng.integers(0, int(ni) + 1, endpoint=True)) for ni in n]
    result = ma_freeman_tukey(x, n)
    assert isinstance(result, dict)
    for key in ("ft", "var", "se", "k"):
        assert key in result
    assert result["k"] == n_studies
    assert len(result["ft"]) == n_studies
    assert len(result["var"]) == n_studies
    assert len(result["se"]) == n_studies
