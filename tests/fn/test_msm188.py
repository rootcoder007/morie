"""Verification tests for msm188.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eqs. 9.15 to 9.26 pp.346-347, the quadratic program. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm188 import qplincon


def test_the_first_illustrative_example_matches_the_printed_solution():
    # minimize x^2 subject to x >= 1: alpha = c / (a'a) = 1, x = 1
    res = qplincon([1.0], 1.0)
    assert res["alpha"] == pytest.approx(1.0, rel=1e-12)
    assert list(res["x"]) == pytest.approx([1.0], rel=1e-12)
    assert res["primal_value"] == pytest.approx(1.0, rel=1e-12)
    # L(alpha) = -alpha^2 + 2 alpha at alpha = 1
    assert res["dual_value"] == pytest.approx(-1.0 + 2.0, rel=1e-12)


def test_the_second_illustrative_example_matches_the_printed_solution():
    # minimize x^2 + y^2 subject to x + y >= 2: x = y = alpha = 1
    res = qplincon([1.0, 1.0], 2.0)
    assert res["alpha"] == pytest.approx(2.0 / 2.0, rel=1e-12)
    assert list(res["x"]) == pytest.approx([1.0, 1.0], rel=1e-12)
    # L(alpha) = -2 alpha^2 + 4 alpha at alpha = 1
    assert res["dual_value"] == pytest.approx(-2.0 + 4.0, rel=1e-12)
    assert res["primal_value"] == pytest.approx(2.0, rel=1e-12)


def test_strong_duality_holds_at_the_optimum():
    for a, c in ([[1.0], 1.0], [[1.0, 1.0], 2.0], [[3.0, 4.0], 10.0]):
        res = qplincon(a, c)
        assert res["dual_value"] == pytest.approx(res["primal_value"],
                                                   rel=1e-12)


def test_the_constraint_is_met_exactly_at_the_optimum():
    res = qplincon([3.0, 4.0], 10.0)
    assert res["constraint"] == pytest.approx(10.0, rel=1e-12)
    assert res["active"] is True
    # alpha = c / (a'a) = 10 / 25
    assert res["alpha"] == pytest.approx(0.4, rel=1e-12)


def test_a_zero_constraint_vector_is_refused():
    with pytest.raises(ValueError):
        qplincon([0.0, 0.0], 1.0)
