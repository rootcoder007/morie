"""Tests for abdpd.abduction_modification_prediction."""

import pytest

from morie.fn.abdpd import abduction_modification_prediction

EQS = {"X": (("u1",), lambda u1: u1), "Y": (("X", "u2"), lambda X, u2: 2 * X + u2)}


def test_abdpd_basic():
    out = abduction_modification_prediction(
        {"X": 1.0, "Y": 3.0}, EQS, ["u1", "u2"], {"X": 4.0}, "Y"
    )
    assert out["abducted"]["u2"] == pytest.approx(1.0, abs=1e-6)
    assert out["factual"] == pytest.approx(3.0, abs=1e-6)
    assert out["counterfactual"] == pytest.approx(9.0, abs=1e-6)


def test_abdpd_edge():
    with pytest.raises(ValueError):
        abduction_modification_prediction({"X": 1.0}, EQS, ["u1", "u2"], {"u1": 0.0}, "Y")
    with pytest.raises(ValueError):
        abduction_modification_prediction({}, EQS, ["u1"], {"X": 1.0}, "Y")  # no evidence
