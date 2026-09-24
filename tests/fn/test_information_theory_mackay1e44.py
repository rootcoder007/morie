"""Verification tests for information_theory_mackay1e44.repcn.

The expected values are recomputed from MacKay (2003) eq. (1.44)-(1.45) p. 17 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay1e44 import repcn


def test_repcn_reproduces_the_books_iteration():
    pb, f, n0, iters = 1e-15, 0.1, 68.0, 3
    n = n0
    denom = math.log10(4.0 * f * (1.0 - f))
    half = None
    for _ in range(iters):
        half = (math.log10(pb) + math.log10(math.sqrt(math.pi * n / 8.0) / f)) / denom
        n = 2.0 * half + 1.0
    res = repcn(pb, f, n0, iters)
    assert res["n"] == pytest.approx(n, rel=1e-12)
    assert res["half"] == pytest.approx(half, rel=1e-12)
    assert res["denom"] == pytest.approx(denom, rel=1e-12)


def test_repcn_needs_a_longer_code_for_a_stricter_target():
    loose = repcn(1e-6, 0.1)["n"]
    strict = repcn(1e-15, 0.1)["n"]
    assert strict > loose


def test_repcn_rejects_a_channel_that_is_not_better_than_chance():
    with pytest.raises(ValueError):
        repcn(1e-6, 0.5)
