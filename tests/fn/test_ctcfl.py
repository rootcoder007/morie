"""Tests for ctcfl.counterfactual_notation."""

import pytest

from morie.fn.ctcfl import counterfactual_notation

EQS = {"X": (("u1",), lambda u1: u1), "Y": (("X", "u2"), lambda X, u2: 3 * X + u2)}


def test_ctcfl_basic():
    out = counterfactual_notation({"u1": 2.0, "u2": 1.0}, EQS, "X", 5.0, "Y")
    assert out["factual"] == pytest.approx(7.0)
    assert out["counterfactual"] == pytest.approx(16.0)
    assert out["effect"] == pytest.approx(9.0)


def test_ctcfl_edge():
    with pytest.raises(ValueError):
        counterfactual_notation({"u1": 1.0}, EQS, "u1", 0.0, "Y")  # X must be endogenous
