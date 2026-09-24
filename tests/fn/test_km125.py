"""Verification tests for km125.

Kamath, Keenan, Somers and Sorenson (2024), the normalised n-gram weight. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km125 import kamath_ch8_ngram_weight


def test_the_ngram_weights_normalise_to_one():
    # f = (1/Z) sum idf over each window
    res = kamath_ch8_ngram_weight([[1.0, 2.0], [3.0, 4.0]])
    assert [round(v, 12) for v in res["weights"]] == [0.3, 0.7]
    assert sum(res["weights"]) == pytest.approx(1.0, rel=1e-12)


def test_equal_windows_share_the_weight_equally():
    res = kamath_ch8_ngram_weight([[1.0, 1.0], [1.0, 1.0]])
    for v in res["weights"]:
        assert v == pytest.approx(0.5, rel=1e-12)


def test_an_explicit_normaliser_is_used_as_given():
    res = kamath_ch8_ngram_weight([[1.0, 2.0], [3.0, 4.0]], Z=20.0)
    assert res["weights"][0] == pytest.approx(3.0 / 20.0, rel=1e-12)
