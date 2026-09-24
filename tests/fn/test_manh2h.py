"""Tests for manh2h.ma_network_node_split."""

import pytest

from morie.fn import _array_core as np

from morie.fn.manh2h import ma_network_node_split


def test_manh2h_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    design = []
    for _ in range(n):
        t1 = int(rng.integers(0, 3))
        t2 = int(rng.integers(0, 3))
        while t2 == t1:
            t2 = int(rng.integers(0, 3))
        design.append([t1, t2])
    if not any((d[0], d[1]) == (0, 1) or (d[0], d[1]) == (1, 0) for d in design):
        design[0] = [0, 1]
    yi = rng.normal(0, 1, n)
    vi = rng.uniform(0.1, 1.0, n)
    edge = [0, 1]
    result = ma_network_node_split(yi, vi, design, edge)
    assert isinstance(result, dict)
    assert "direct" in result
    assert "indirect" in result
    assert "z" in result
    assert "p" in result


def test_manh2h_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    design = []
    for _ in range(n):
        t1 = int(rng.integers(0, 3))
        t2 = int(rng.integers(0, 3))
        while t2 == t1:
            t2 = int(rng.integers(0, 3))
        design.append([t1, t2])
    yi = rng.normal(0, 1, n)
    vi = [-0.1] * n
    edge = [0, 1]
    with pytest.raises(ValueError, match="sampling variances"):
        ma_network_node_split(yi, vi, design, edge)
