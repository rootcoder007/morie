"""Anchored tests for meteor (Banerjee-Lavie 2005).

Anchor: the paper's own worked example (Section 2.1): candidate
"the president spoke to the audience" vs reference "the president
then spoke to the audience" has exactly two chunks ("the president"
and "spoke to the audience") with all six candidate unigrams
matched, so P = 1, R = 6/7, Fmean = 10PR/(R + 9P),
Penalty = 0.5 (2/6)^3, Score = Fmean (1 - Penalty).
"""

from morie.fn.meteor import meteor, meteor_score


def test_meteor_paper_example():
    res = meteor("the president spoke to the audience",
                 "the president then spoke to the audience")
    assert res["matches"] == 6
    assert res["chunks"] == 2
    p, r = 1.0, 6.0 / 7.0
    fmean = 10.0 * p * r / (r + 9.0 * p)
    pen = 0.5 * (2.0 / 6.0) ** 3
    assert abs(res["precision"] - p) < 1e-15
    assert abs(res["recall"] - r) < 1e-15
    assert abs(res["fmean"] - fmean) < 1e-12
    assert abs(res["penalty"] - pen) < 1e-15
    assert abs(res["score"] - fmean * (1.0 - pen)) < 1e-12


def test_meteor_perfect_match_single_chunk():
    res = meteor("a b c d", "a b c d")
    assert res["chunks"] == 1
    assert res["matches"] == 4
    # P = R = 1 -> Fmean = 1; Penalty = 0.5 (1/4)^3
    assert abs(res["fmean"] - 1.0) < 1e-15
    assert abs(res["score"] - (1.0 - 0.5 / 64.0)) < 1e-15


def test_meteor_no_match_scores_zero():
    res = meteor("x y z", "a b c")
    assert res["score"] == 0.0
    assert res["matches"] == 0


def test_meteor_max_fragmentation():
    # every match its own chunk: penalty -> 0.5 (m/m)^3 = 0.5
    res = meteor("a x b", "b y a")
    assert res["matches"] == 2
    assert res["chunks"] == 2
    assert abs(res["penalty"] - 0.5) < 1e-15
    assert meteor_score is meteor
