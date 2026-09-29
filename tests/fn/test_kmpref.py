"""Tests for kmpref.kamath_prefix_tuning."""

from morie.fn import _array_core as np
from morie.fn.kmpref import kamath_prefix_tuning


def test_kmpref_basic():
    """Test basic functionality."""
    prefix_K = np.random.default_rng(42).normal(0, 1, 100)
    prefix_V = np.random.default_rng(42).normal(0, 1, 100)
    K_input = np.random.default_rng(42).normal(0, 1, 100)
    V_input = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_prefix_tuning(prefix_K, prefix_V, K_input, V_input)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmpref_edge():
    """Test edge cases."""
    prefix_K = np.random.default_rng(42).normal(0, 1, 100)
    prefix_V = np.random.default_rng(42).normal(0, 1, 100)
    K_input = np.random.default_rng(42).normal(0, 1, 100)
    V_input = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_prefix_tuning(prefix_K, prefix_V, K_input, V_input)
    assert isinstance(result, dict)


def test_prefix_mass_recomputed():
    import math

    import pytest

    PK, PV = [[1.0, 0.0], [0.0, 2.0]], [[5.0], [6.0]]
    K, V = [[0.5, 0.5]], [[1.0]]
    q = [1.0, 1.0]
    keys = PK + K
    s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(2) for k in keys]
    e = [math.exp(v - max(s)) for v in s]
    w = [v / sum(e) for v in e]
    r = kamath_prefix_tuning(PK, PV, K, V, Q=[q])
    assert r["prefix_attention_mass"][0] == pytest.approx(w[0] + w[1], rel=1e-13)
    assert r["attention_output"][0][0] == pytest.approx(5 * w[0] + 6 * w[1] + w[2], rel=1e-13)
    assert r["n_trainable"] == 6
