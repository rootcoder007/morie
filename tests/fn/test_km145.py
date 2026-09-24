"""Verification tests for km145.

Kamath, Keenan, Somers and Sorenson (2024), the multimodal autoregressive loss. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km145 import kamath_ch9_mmllm_autoregressive


def test_the_autoregressive_loss_sums_the_response_log_probabilities():
    # L = -sum_i log p(R_i | I, R_<i)
    res = kamath_ch9_mmllm_autoregressive([0.5, 0.25], None)
    assert res["estimate"] == pytest.approx(
        -(math.log(0.5) + math.log(0.25)), rel=1e-12)


def test_a_certain_response_costs_nothing():
    res = kamath_ch9_mmllm_autoregressive([1.0, 1.0], None)
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_a_longer_response_accumulates_more_loss():
    short = kamath_ch9_mmllm_autoregressive([0.5], None)["estimate"]
    long_ = kamath_ch9_mmllm_autoregressive([0.5, 0.5], None)["estimate"]
    assert long_ > short
