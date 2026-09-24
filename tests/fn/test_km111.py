"""Verification tests for km111.

Kamath, Keenan, Somers and Sorenson (2024), Large Language Models: A
Deep Dive, the RAG faithfulness metric. Expected values are recomputed in the test body.
"""

import math

import pytest

from morie.fn.km111 import kamath_ch7_faithfulness_metric


def test_faithfulness_is_the_supported_fraction():
    # the RAG faithfulness metric: supported facts over total facts
    facts = [1, 1, 0, 1]
    res = kamath_ch7_faithfulness_metric(facts)
    assert res["estimate"] == pytest.approx(3.0 / 4.0, rel=1e-12)
    assert res["n_supported"] == 3
    assert res["n_facts"] == 4


def test_every_fact_supported_scores_one():
    assert kamath_ch7_faithfulness_metric([1, 1, 1])["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_no_fact_supported_scores_zero():
    assert kamath_ch7_faithfulness_metric([0, 0, 0])["estimate"] == pytest.approx(0.0, abs=1e-15)
