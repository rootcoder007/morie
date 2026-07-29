"""Tests for b108.burkov_lm_ch1_logistic_regression."""

from morie.fn.b108 import burkov_lm_ch1_logistic_regression


def test_b108_basic():
    assert burkov_lm_ch1_logistic_regression([0.0], [5.0], 0.0)["estimate"] == 0.5


def test_b108_edge():
    out = burkov_lm_ch1_logistic_regression([10.0], [1.0], 0.0)
    assert out["predicted_class"] == 1
