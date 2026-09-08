"""Tests for causshap.causal_shap_decomposition."""

import pytest

from morie.fn.causshap import causal_shap_decomposition

W = {"a": 3.0, "b": 2.0, "c": -1.0}


def _v(S):
    return sum(W[f] for f in S)


def test_causshap_basic():
    out = causal_shap_decomposition(_v, ["a", "b", "c"])
    assert out["shapley"] == pytest.approx(W)
    assert out["efficiency_gap"] == pytest.approx(0.0, abs=1e-12)


def test_causshap_edge():
    def v2(S):
        return 1.0 if set(S) == {"a", "b"} else 0.0

    sym = causal_shap_decomposition(v2, ["a", "b"])
    assert sym["shapley"]["a"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        causal_shap_decomposition(_v, [])
