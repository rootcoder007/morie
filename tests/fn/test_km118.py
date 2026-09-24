"""Verification tests for km118.

Kamath, Keenan, Somers and Sorenson (2024), eq. 8.6, ROUGE-N recall. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km118 import kamath_ch8_rouge_n


def test_rouge_n_is_matched_over_reference_ngrams():
    # Eq 8.6: ROUGE-N = sum Count_match / sum Count over the references
    refs = [["the", "cat", "sat", "down"]]
    cand = ["the", "cat", "stood"]
    res = kamath_ch8_rouge_n(refs, 2, candidate=cand)
    ref_bigrams = [("the", "cat"), ("cat", "sat"), ("sat", "down")]
    cand_bigrams = [("the", "cat"), ("cat", "stood")]
    matched = sum(1 for g in ref_bigrams if g in cand_bigrams)
    assert res["total_reference_ngrams"] == len(ref_bigrams)
    assert res["matched"] == matched
    assert res["estimate"] == pytest.approx(matched / len(ref_bigrams), rel=1e-12)


def test_a_candidate_equal_to_the_reference_has_full_recall():
    refs = [["a", "b", "c"]]
    res = kamath_ch8_rouge_n(refs, 2, candidate=["a", "b", "c"])
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_a_disjoint_candidate_has_zero_recall():
    refs = [["a", "b", "c"]]
    res = kamath_ch8_rouge_n(refs, 2, candidate=["x", "y", "z"])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_the_candidate_is_required_because_rouge_compares_two_texts():
    with pytest.raises(ValueError):
        kamath_ch8_rouge_n([["a", "b"]], 2)
