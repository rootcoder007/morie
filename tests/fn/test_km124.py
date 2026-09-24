"""Verification tests for km124.

Kamath, Keenan, Somers and Sorenson (2024), the n-gram idf embedding. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km124 import kamath_ch8_ngram_embedding


def test_the_ngram_embedding_sums_idf_over_the_window():
    # E(x_i^n) = sum_{k=i}^{i+n-1} idf(x_k)
    idf = [1.0, 2.0, 3.0]
    res = kamath_ch8_ngram_embedding(idf, 1, 2)
    assert res["estimate"] == pytest.approx(2.0 + 3.0, rel=1e-12)


def test_a_unigram_window_is_the_single_idf_value():
    res = kamath_ch8_ngram_embedding([1.0, 2.0, 3.0], 0, 1)
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_a_window_past_the_end_is_refused():
    with pytest.raises((ValueError, IndexError)):
        kamath_ch8_ngram_embedding([1.0, 2.0], 1, 3)
