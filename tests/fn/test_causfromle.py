"""Tests for causfromle.causal_e_value."""

import pytest

from morie.fn.causfromle import causal_e_value


def test_causfromle_basic():
    # E = RR + sqrt(RR*(RR-1)); at RR = 2 that is 2 + sqrt(2) = 3.4142
    out = causal_e_value(2.0)
    assert out["evalue"] == pytest.approx(2 + 2**0.5)
    assert out["rr"] == pytest.approx(2.0)
    # RR = 1 means no association: the E-value bottoms out at 1
    assert causal_e_value(1.0)["evalue"] == pytest.approx(1.0)


def test_causfromle_edge():
    # a protective RR is inverted before the formula, so 0.5 and 2 agree
    assert causal_e_value(0.5)["evalue"] == pytest.approx(causal_e_value(2.0)["evalue"])
    with pytest.raises(ValueError):
        causal_e_value(0.0)
