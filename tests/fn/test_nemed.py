"""Tests for nemed.nested_counterfactual_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.nemed import nested_counterfactual_mediation


def test_nemed_basic():
    rng = np.random.default_rng(42)
    n = 6000
    x = (rng.random(n) < 0.5).astype(float)
    m = 0.9 * x + rng.normal(scale=0.7, size=n)
    y = 0.4 * x + 1.0 * m + 0.5 * x * m + rng.normal(scale=0.7, size=n)
    out = nested_counterfactual_mediation(x, m, y)
    assert out["nde"] == pytest.approx(0.4, abs=0.1)
    assert out["nie"] == pytest.approx(1.5 * 0.9, abs=0.1)
    assert out["te"] == pytest.approx(out["nde"] + out["nie"])


def test_nemed_edge():
    with pytest.raises(ValueError):
        nested_counterfactual_mediation([1.0] * 5, [1.0] * 5, [1.0] * 5)
