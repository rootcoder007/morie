"""Tests for morie.fn.causfromle: the E-value closed form."""

import math

from morie.fn.causfromle import causal_e_value


def test_closed_form():
    for rr in (2.0, 0.5, 1.0, 3.7):
        s = max(rr, 1 / rr)
        assert abs(causal_e_value(rr)["evalue"] - (s + math.sqrt(s * (s - 1)))) < 1e-14
