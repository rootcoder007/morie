"""Verification tests for km078.

Kamath, Keenan, Somers and Sorenson (2024), eq. 6.2, the alignment function as a type. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km078 import kamath_ch6_alignment_function


def test_the_alignment_function_is_checked_against_its_output_space():
    # Eq 6.2 is a type: f: (a, b) -> y in the declared space
    res = kamath_ch6_alignment_function("the cat sat", "a cat sat", "3way", f=lambda a, b: "ALIGNED")
    assert res["label"] == "ALIGNED"
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_a_regression_head_returns_its_own_score():
    assert kamath_ch6_alignment_function("x", "y", "reg", f=lambda a, b: 0.25)["estimate"] == \
        pytest.approx(0.25, rel=1e-12)


def test_a_label_outside_the_declared_space_is_refused():
    with pytest.raises(ValueError):
        kamath_ch6_alignment_function("x", "y", "3way", f=lambda a, b: "SOMETHING_ELSE")
