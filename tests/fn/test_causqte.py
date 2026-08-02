"""Tests for causqte.causal_quantile_treatment_effect."""

from morie.fn import _array_core as np
import pytest

from morie.fn.causqte import causal_quantile_treatment_effect


def test_causqte_basic():
    # randomised (ps = 0.5), constant shift of 2 -> QTE(tau) = 2
    rng = np.random.default_rng(42)
    n = 4000
    T = (rng.random(n) < 0.5).astype(float)
    y = rng.normal(size=n) + 2.0 * T
    result = causal_quantile_treatment_effect(y, T, np.full(n, 0.5), tau=0.5)
    assert result["qte"] == pytest.approx(2.0, abs=0.15)  # measured ~2.03


def test_causqte_edge():
    with pytest.raises(ValueError):
        causal_quantile_treatment_effect([1.0, 2.0], [1, 0], [0.5, 0.5], tau=1.5)
    with pytest.raises(ValueError):
        causal_quantile_treatment_effect([1.0, 2.0], [1, 1], [0.5, 0.5], tau=0.5)
