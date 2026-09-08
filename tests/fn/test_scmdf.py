"""Tests for scmdf.scm_definition."""

import pytest

from morie.fn.scmdf import scm_definition


def test_scmdf_basic():
    out = scm_definition(
        {"u": 1.0},
        {"X": (("u",), lambda u: 2 * u), "Y": (("X", "u"), lambda X, u: X + u)},
    )
    assert out["values"]["Y"] == pytest.approx(3.0)
    assert out["order"] == ["X", "Y"]


def test_scmdf_edge():
    with pytest.raises(ValueError):
        scm_definition({}, {"A": (("B",), lambda B: B), "B": (("A",), lambda A: A)})  # cycle
    with pytest.raises(ValueError):
        scm_definition({"u": 1.0}, {"X": (("w",), lambda w: w)})  # unknown parent
