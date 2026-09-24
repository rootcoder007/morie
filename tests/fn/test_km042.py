"""Verification tests for km042.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.1, the class probability as its answer word's probability. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km042 import kamath_ch3_prompt_label_mapping


def test_the_class_probability_is_its_answer_words_probability():
    # Eq 3.1: p(y|x) = p(z = M(y) | x')
    dist = {"great": 0.7, "terrible": 0.3}
    M = {"positive": "great", "negative": "terrible"}
    for label, word in M.items():
        res = kamath_ch3_prompt_label_mapping(dist, label, M)
        assert res["estimate"] == pytest.approx(dist[word], rel=1e-12)


def test_the_mapping_selects_which_answer_word_counts():
    dist = {"great": 0.7, "terrible": 0.3}
    swapped = {"positive": "terrible", "negative": "great"}
    res = kamath_ch3_prompt_label_mapping(dist, "positive", swapped)
    assert res["estimate"] == pytest.approx(0.3, rel=1e-12)


def test_an_empty_distribution_is_refused():
    with pytest.raises(ValueError):
        kamath_ch3_prompt_label_mapping({}, "positive", {"positive": "great"})
