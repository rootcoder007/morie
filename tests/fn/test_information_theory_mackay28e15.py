"""Verification tests for information_theory_mackay28e15.msglen.

The expected values are recomputed from MacKay (2003) eq. (28.15) p. 352 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay28e15 import msglen


def test_msglen_inverts_between_probability_and_bits():
    res = msglen(p=0.125)
    assert res["length"] == pytest.approx(3.0, abs=1e-12)
    back = msglen(length=3.0)
    assert back["p"] == pytest.approx(0.125, abs=1e-12)
    assert res["nats"] == pytest.approx(3.0 * math.log(2.0), abs=1e-12)


def test_msglen_certainty_costs_no_bits():
    assert msglen(p=1.0)["length"] == pytest.approx(0.0, abs=1e-12)


def test_msglen_needs_exactly_one_of_its_two_arguments():
    with pytest.raises(ValueError):
        msglen()
    with pytest.raises(ValueError):
        msglen(p=0.5, length=1.0)
