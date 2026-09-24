"""Tests for otmarsh.ot_marginal_shift."""

import math

from morie.fn import _array_core as np

from morie.fn.otmarsh import ot_marginal_shift


def test_otmarsh_basic():
    """Test basic functionality with a scalar delta."""
    rng = np.random.default_rng(44)
    n, m = 40, 30
    a = list(rng.uniform(0, 1, n))
    b = list(rng.uniform(0, 1, m))
    # Normalise target to the source total so partial transport is feasible.
    sa = sum(a)
    sb = sum(b)
    b = [bi * sa / sb for bi in b]
    C = [list(rng.uniform(0, 1, m)) for _ in range(n)]
    delta = 0.5
    result = ot_marginal_shift(a, b, C, delta)
    assert isinstance(result, dict)
    assert "T" in result
    assert "cost" in result
    assert "a_shift" in result
    assert "removed" in result
    assert "mass" in result
    assert "n" in result
    assert "m" in result
    assert len(result["T"]) == n
    assert len(result["T"][0]) == m
    assert result["n"] == n
    assert result["m"] == m
    assert math.isfinite(result["cost"])


def test_otmarsh_edge():
    """Test edge case with a per-bin delta vector."""
    rng = np.random.default_rng(42)
    n, m = 20, 25
    a = list(rng.uniform(0, 1, n))
    b = list(rng.uniform(0, 1, m))
    # Normalise target to the source total so partial transport is feasible.
    sa = sum(a)
    sb = sum(b)
    b = [bi * sa / sb for bi in b]
    C = [list(rng.uniform(0, 1, m)) for _ in range(n)]
    delta = list(rng.uniform(0, 0.01, n))
    result = ot_marginal_shift(a, b, C, delta)
    assert isinstance(result, dict)
    assert "T" in result
    assert "cost" in result
    assert "a_shift" in result
    assert len(result["T"]) == n
    assert len(result["T"][0]) == m
    assert result["n"] == n
    assert result["m"] == m
    assert math.isfinite(result["cost"])
