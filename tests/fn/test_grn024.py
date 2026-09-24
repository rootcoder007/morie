"""Verification tests for grn024.

Geron (2023), *Hands-On Machine Learning with Scikit-Learn, Keras and
TensorFlow*, 3rd edition, ch 4, eq. 4.23, the cross entropy gradient. Expected values are recomputed in
the test body and the docstring's own worked value is asserted too.
"""

import math

import pytest

from morie.fn.grn024 import geron_ch4_cross_entropy_gradient_vector


def test_the_gradient_is_the_mean_error_times_the_features():
    # a uniform model over three classes under-predicts the true one by
    # 1/3 - 1 = -2/3
    res = geron_ch4_cross_entropy_gradient_vector([[1.0]], [0], [[0.0, 0.0, 0.0]], k=0)
    assert list(res["gradient"]) == pytest.approx([1.0 / 3.0 - 1.0],
                                                   rel=1e-9)
    assert [round(v, 6) for v in res["gradient"]] == pytest.approx(
        [-0.666667], abs=1e-6)
    assert round(res["mean_error"], 6) == pytest.approx(-0.666667,
                                                         abs=1e-6)


def test_a_class_nobody_belongs_to_gets_a_positive_gradient():
    res = geron_ch4_cross_entropy_gradient_vector([[1.0]], [0], [[0.0, 0.0, 0.0]], k=1)
    assert res["gradient"][0] == pytest.approx(1.0 / 3.0, rel=1e-9)
    assert round(res["gradient"][0], 6) == pytest.approx(0.333333,
                                                          abs=1e-6)


def test_the_gradients_over_all_classes_cancel_out():
    # every row's probabilities sum to one and exactly one label is set,
    # so the errors across classes must sum to zero
    total = sum(geron_ch4_cross_entropy_gradient_vector([[1.0]], [0], [[0.0, 0.0, 0.0]], k=k)["gradient"][0]
                for k in range(3))
    assert total == pytest.approx(0.0, abs=1e-9)


def test_a_confident_correct_model_asks_for_almost_no_change():
    res = geron_ch4_cross_entropy_gradient_vector([[1.0]], [0], [[50.0, 0.0, 0.0]], k=0)
    assert abs(res["gradient"][0]) < 1e-9


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grn024 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
