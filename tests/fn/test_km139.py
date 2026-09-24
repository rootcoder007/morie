"""Verification tests for km139.

Kamath, Keenan, Somers and Sorenson (2024), the SimVLM prefix language-modelling loss. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km139 import kamath_ch9_simvlm_prefixlm


def test_the_prefix_loss_scores_only_the_suffix():
    # L_PrefixLM = -log P(x_{>=T_p} | x_{<T_p})
    res = kamath_ch9_simvlm_prefixlm(None, [0.5, 0.5, 0.25], 1)
    # the first token is the prefix: 0.5 * 0.25 = 1/8
    assert res["estimate"] == pytest.approx(math.log(8.0), rel=1e-12)


def test_a_longer_prefix_leaves_less_to_predict():
    short = kamath_ch9_simvlm_prefixlm(None, [0.5, 0.5, 0.25], 1)["estimate"]
    long_ = kamath_ch9_simvlm_prefixlm(None, [0.5, 0.5, 0.25], 2)["estimate"]
    assert long_ < short


def test_scoring_the_whole_sequence_uses_every_token():
    res = kamath_ch9_simvlm_prefixlm(None, [0.5, 0.5], 0)
    assert res["estimate"] == pytest.approx(math.log(4.0), rel=1e-12)
