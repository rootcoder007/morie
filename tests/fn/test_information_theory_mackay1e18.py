"""Verification tests for information_theory_mackay1e18.r3post.

The expected values are recomputed from MacKay (2003) eq. (1.18) p. 9 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay1e18 import r3post


def test_r3post_matches_the_book_posterior():
    f = 0.1
    r = [1, 1, 0]
    # (1.18): P(s | r) proportional to prod_n f^[r_n != s] (1-f)^[r_n == s]
    lik = {}
    for s in (0, 1):
        p = 1.0
        for b in r:
            p *= f if b != s else (1.0 - f)
        lik[s] = p
    total = lik[0] + lik[1]
    res = r3post(r, f)
    assert res["p1"] == pytest.approx(lik[1] / total, abs=1e-12)
    assert res["p0"] == pytest.approx(lik[0] / total, abs=1e-12)
    assert res["decoded"] == 1
    # the likelihood ratio per agreeing bit, gamma = (1-f)/f
    assert res["gamma"] == pytest.approx((1.0 - f) / f, abs=1e-12)


def test_r3post_is_symmetric_under_bit_flip():
    f = 0.23
    a = r3post([1, 0, 1], f)
    b = r3post([0, 1, 0], f)
    assert a["p1"] == pytest.approx(b["p0"], abs=1e-12)
    assert a["decoded"] == 1 and b["decoded"] == 0


def test_r3post_rejects_input_that_is_not_three_bits():
    with pytest.raises(ValueError):
        r3post([1, 1], 0.1)
    with pytest.raises(ValueError):
        r3post([1, 1, 1], 0.0)
