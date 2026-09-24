"""Verification tests for km150.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.22, the Flamingo dataset mixture objective. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km150 import kamath_ch9_flamingo_dataset_mix


def test_the_dataset_mixture_weights_each_corpus_loss():
    # Eq 9.22: sum_m lambda_m E[-sum_l log p]
    res = kamath_ch9_flamingo_dataset_mix([[[0.5]], [[0.25]]], [0.25, 0.75])
    expected = 0.25 * math.log(2.0) + 0.75 * math.log(4.0)
    assert res["estimate"] == pytest.approx(expected, rel=1e-10)
    assert round(res["estimate"], 6) == pytest.approx(1.213008, abs=1e-6)


def test_all_weight_on_one_corpus_returns_its_own_loss():
    res = kamath_ch9_flamingo_dataset_mix([[[0.5]], [[0.25]]], [1.0, 0.0])
    assert res["estimate"] == pytest.approx(math.log(2.0), rel=1e-12)


def test_a_certain_model_on_every_corpus_costs_nothing():
    res = kamath_ch9_flamingo_dataset_mix([[[1.0]], [[1.0]]], [0.5, 0.5])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
