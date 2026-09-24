"""Tests for km094.kamath_ch6_debias_regularizer."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.km094 import kamath_ch6_debias_regularizer


def test_km094_basic():
    rng = np.random.default_rng(42)
    words = ["man", "woman", "king", "queen"]
    dim = 3
    E = {w: rng.normal(0, 1, dim) for w in words}
    A = [("man", "woman"), ("king", "queen"), ("man", "king")]
    lam = 0.5
    result = kamath_ch6_debias_regularizer(A, E, lam)
    assert isinstance(result, dict)
    expected_keys = {"estimate", "per_pair", "unweighted", "lam", "n", "method"}
    assert expected_keys.issubset(result.keys())
    assert result["n"] == len(A)
    assert result["lam"] == lam
    assert isinstance(result["per_pair"], list)
    assert len(result["per_pair"]) == len(A)
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["unweighted"])
    assert result["estimate"] == pytest.approx(lam * result["unweighted"])


def test_km094_edge():
    rng = np.random.default_rng(42)
    E = {"a": rng.normal(0, 1, 2), "b": rng.normal(0, 1, 2)}
    A = [("a", "b")]
    with pytest.raises(ValueError):
        kamath_ch6_debias_regularizer(A, E, -1.0)
