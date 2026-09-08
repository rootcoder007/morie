"""Tests for caussc.causal_synthetic_control."""

from morie.fn import _array_core as np
import pytest

from morie.fn.caussc import causal_synthetic_control


def test_caussc_basic():
    # x1 an exact convex combination of the donors
    rng = np.random.default_rng(42)
    X0 = rng.normal(size=(6, 4))
    w_true = np.array([0.5, 0.3, 0.2, 0.0])
    out = causal_synthetic_control(X0 @ w_true, X0)
    assert out["weights"] == pytest.approx(w_true, abs=0.02)
    assert out["rmse_pre"] == pytest.approx(0.0, abs=1e-3)


def test_caussc_edge():
    out = causal_synthetic_control([0.3, 0.7], np.eye(2))
    assert out["weights"].sum() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        causal_synthetic_control([1.0, 2.0], np.ones((3, 2)))  # k mismatch
    with pytest.raises(ValueError):
        causal_synthetic_control([1.0], np.ones((1, 1)))  # 1 donor
