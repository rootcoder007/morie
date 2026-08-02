"""Tests for causscss.causal_synthetic_subset."""

from morie.fn import _array_core as np
import pytest

from morie.fn.causscss import causal_synthetic_subset


def test_causscss_basic():
    rng = np.random.default_rng(42)
    X0 = rng.normal(size=(30, 12))
    x1 = 0.6 * X0[:, 3] + 0.4 * X0[:, 7] + rng.normal(scale=0.02, size=30)
    result = causal_synthetic_subset(x1, X0, lam=0.05)
    assert {3, 7}.issubset(set(result["support"]))
    assert result["weights"][3] == pytest.approx(0.6, abs=0.1)
    assert result["weights"][7] == pytest.approx(0.4, abs=0.1)
    assert result["weights"].sum() == pytest.approx(1.0)


def test_causscss_edge():
    with pytest.raises(ValueError):
        causal_synthetic_subset([1.0, 2.0], np.ones((3, 2)), lam=0.1)  # k mismatch
    with pytest.raises(ValueError):
        causal_synthetic_subset([1.0, 2.0], np.ones((2, 3)), lam=-0.1)
