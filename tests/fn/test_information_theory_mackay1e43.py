"""Verification tests for information_theory_mackay1e43.repcpb.

The expected values are recomputed from MacKay (2003) eq. (1.42)-(1.43) p. 17 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay1e43 import repcpb


def test_repcpb_leading_term_and_approximation():
    n, f = 61, 0.1
    k = (n + 1) // 2
    leading = math.comb(n, k) * f ** k * (1.0 - f) ** (n - k)
    # (1.43): p_b approximately (pi N / 8)^(-1/2) f (4 f (1-f))^((N-1)/2)
    approx2 = (1.0 / math.sqrt(math.pi * n / 8.0)) * f * (4.0 * f * (1.0 - f)) ** ((n - 1) / 2.0)
    res = repcpb(n, f)
    assert res["leading"] == pytest.approx(leading, rel=1e-12)
    assert res["approx2"] == pytest.approx(approx2, rel=1e-12)


def test_repcpb_error_falls_as_the_blocklength_grows():
    f = 0.1
    errs = [repcpb(n, f)["leading"] for n in (3, 11, 31, 61)]
    assert all(b < a for a, b in zip(errs, errs[1:]))


def test_repcpb_rejects_an_even_blocklength():
    with pytest.raises(ValueError):
        repcpb(4, 0.1)
