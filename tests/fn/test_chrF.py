"""Tests for chrF (character n-gram F-score).

Replaces the generated stub, which imported ``chrf``.
"""

from morie.fn.chrF import chrf_score


def test_identical_strings_score_one():
    res = chrf_score("the cat sat", "the cat sat")
    assert abs(res["chrf"] - 1.0) < 1e-12
    assert abs(res["chrP"] - 1.0) < 1e-12
    assert abs(res["chrR"] - 1.0) < 1e-12
    assert all(abs(o["precision"] - 1.0) < 1e-12 for o in res["per_order"])


def test_disjoint_strings_score_zero():
    assert chrf_score("aaaa", "bbbb")["chrf"] == 0.0


def test_partial_overlap_lies_between():
    score = chrf_score("the cat sat", "the cat ran")["chrf"]
    assert 0.0 < score < 1.0


def test_beta_weights_recall_over_precision():
    # a short hypothesis against a long reference: high precision, low
    # recall, so a larger beta (more weight on recall) must not increase
    # the score
    hyp, ref = "the cat", "the cat sat on the mat"
    low = chrf_score(hyp, ref, beta=0.5)["chrf"]
    high = chrf_score(hyp, ref, beta=3.0)["chrf"]
    assert high < low


def test_per_order_precisions_are_reported_for_each_n():
    res = chrf_score("abcdef", "abcdef", n_char=4)
    assert len(res["per_order"]) == 4
    assert res["n_char"] == 4


def test_multiple_references_take_the_best():
    single = chrf_score("the cat sat", "a dog barked")["chrf"]
    multi = chrf_score("the cat sat",
                       ["a dog barked", "the cat sat"])["chrf"]
    assert multi > single
    assert abs(multi - 1.0) < 1e-12


def test_validation():
    for call in (lambda: chrf_score("a", "b", n_char=0),
                 lambda: chrf_score("a", "b", beta=0.0),
                 lambda: chrf_score("a", "b", word_order=-1)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
