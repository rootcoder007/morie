"""Verification tests for km052.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.11, the T5 template-generation objective. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km052 import kamath_ch3_t5_template_obj


def test_the_t5_objective_sums_the_log_probability_of_the_template():
    # Eq 3.11: sum over the training pairs of log P_T5(T | T(x_in, y))
    data = [("in1", "y1"), ("in2", "y2")]
    P = lambda t, filled: 0.5
    res = kamath_ch3_t5_template_obj(data, "T", P)
    assert res["estimate"] == pytest.approx(2.0 * math.log(0.5), rel=1e-12)
    for v in res["per_example"]:
        assert v == pytest.approx(math.log(0.5), rel=1e-12)


def test_a_certain_template_contributes_no_loss():
    res = kamath_ch3_t5_template_obj([("a", "b")], "T", lambda t, filled: 1.0)
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_a_probability_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        kamath_ch3_t5_template_obj([("a", "b")], "T", lambda t, filled: 1.5)
