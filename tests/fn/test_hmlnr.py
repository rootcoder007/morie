"""Tests for hmlnr.geron_layer_norm_rnn."""

from morie.fn import _array_core as np
from morie.fn.hmlnr import geron_layer_norm_rnn


def test_hmlnr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    beta = 0.8
    result = geron_layer_norm_rnn(x, gamma, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmlnr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    beta = 0.8
    result = geron_layer_norm_rnn(x, gamma, beta)
    assert isinstance(result, dict)


def test_layer_norm_then_activation_recomputed():
    import math

    import pytest

    x = [[1.0, 3.0, 2.0], [0.5, -1.0, 4.0]]
    out = []
    for row in x:
        m = sum(row) / 3
        v = sum((t - m) ** 2 for t in row) / 3
        out.append([math.tanh(2.0 * (t - m) / math.sqrt(v + 1e-3) + 0.5) for t in row])
    r = geron_layer_norm_rnn(x, gamma=2.0, beta=0.5, eps=1e-3)
    for i in range(2):
        assert [float(v) for v in r["h"][i]] == pytest.approx(out[i], rel=1e-13)
