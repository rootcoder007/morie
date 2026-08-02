"""Tests for spwgts.spline_weights."""

from morie.fn import _array_core as np
import pytest

from morie.fn.spwgts import spline_weights


def test_spwgts_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(size=1000)
    e = np.clip(1 / (1 + np.exp(-x)), 0.05, 0.95)
    A = (rng.random(1000) < e).astype(float)
    result = spline_weights(A, x)
    w, eh = result["weights"], result["propensity"]
    # HT identity: w = A/e + (1-A)/(1-e) for the fitted e
    assert w == pytest.approx(A / eh + (1 - A) / (1 - eh))
    assert 0 < result["ess"] <= 1000.0
    assert np.all((eh >= 0.01) & (eh <= 0.99))  # Cole-Hernan truncation


def test_spwgts_edge():
    with pytest.raises(ValueError):
        spline_weights([0.5, 1.0], [1.0, 2.0])  # non-binary A
    with pytest.raises(ValueError):
        spline_weights([1, 0], [1.0, 2.0], knots=[1.0, 1.0])  # <3 distinct knots
