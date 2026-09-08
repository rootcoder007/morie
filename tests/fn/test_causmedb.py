"""Tests for causmedb.causal_mediation_baron_kenny."""

from morie.fn import _array_core as np
import pytest

from morie.fn.causmedb import causal_mediation_baron_kenny


def _dgp(seed=42, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(scale=0.7, size=n)
    y = 0.7 * x + 1.5 * m + rng.normal(scale=0.7, size=n)
    return x, m, y


def test_causmedb_basic():
    x, m, y = _dgp()
    out = causal_mediation_baron_kenny(x, m, y)
    assert out["a"] == pytest.approx(0.8, abs=0.1)
    assert out["b"] == pytest.approx(1.5, abs=0.1)
    assert out["indirect"] == pytest.approx(1.2, abs=0.15)  # a * b
    assert out["c"] == pytest.approx(out["c_prime"] + out["indirect"], abs=1e-6)


def test_causmedb_edge():
    x, m, y = _dgp()
    out = causal_mediation_baron_kenny(x, m, y)
    assert out["proportion_mediated"] == pytest.approx(out["indirect"] / out["c"], abs=1e-6)
    with pytest.raises(ValueError):
        causal_mediation_baron_kenny(x[:10], m, y)  # length mismatch
