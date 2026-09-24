"""Verification tests for information_theory_mackay28e5.postgapx.

The expected values are recomputed from MacKay (2003) eq. (28.5) p. 344 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay28e5 import postgapx


def test_postgapx_is_the_quadratic_form_and_its_gaussian_ratio():
    dw = [0.5, -0.25]
    a = [[4.0, 0.0], [0.0, 2.0]]
    quad = 4.0 * 0.5 ** 2 + 2.0 * 0.25 ** 2
    res = postgapx(dw, a)
    assert res["quadform"] == pytest.approx(quad, rel=1e-12)
    assert res["logratio"] == pytest.approx(-0.5 * quad, rel=1e-12)
    assert res["ratio"] == pytest.approx(math.exp(-0.5 * quad), rel=1e-12)


def test_postgapx_error_bars_are_the_root_diagonal_of_the_inverse_hessian():
    res = postgapx([0.0, 0.0], [[4.0, 0.0], [0.0, 25.0]])
    assert res["errorbars"][0] == pytest.approx(0.5, rel=1e-9)
    assert res["errorbars"][1] == pytest.approx(0.2, rel=1e-9)
    assert res["ratio"] == pytest.approx(1.0, abs=1e-12)


def test_postgapx_requires_a_hessian_matching_the_displacement():
    with pytest.raises(ValueError):
        postgapx([0.0, 0.0], [[1.0]])
