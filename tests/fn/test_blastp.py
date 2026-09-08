"""Tests for blastp (MSP score with Karlin-Altschul E-value).

Replaces the generated stub, which imported ``blast_protein``.
"""

import math

from morie.fn.blastp import blastp


def test_identical_sequences_score_the_whole_length():
    seq = "ACDEFGHIKL"
    res = blastp(seq, seq, match=1.0, mismatch=-1.0)
    assert res["score"] == float(len(seq))
    assert res["length"] == len(seq)
    assert res["q_start"] == 0 and res["s_start"] == 0


def test_the_msp_is_the_best_local_segment():
    # a planted 6-residue match inside otherwise mismatching sequences
    q = "WWWWACDEFGWWWW"
    s = "YYYYACDEFGYYYY"
    res = blastp(q, s, match=1.0, mismatch=-1.0)
    assert res["score"] == 6.0
    assert q[res["q_start"]:res["q_start"] + res["length"]] == "ACDEFG"


def test_e_value_follows_karlin_altschul():
    q, s = "ACDEFGHIKL", "ACDEFGHIKL"
    res = blastp(q, s, match=1.0, mismatch=-1.0, K=0.1, lam=1.0)
    want = 0.1 * len(q) * len(s) * math.exp(-1.0 * res["score"])
    assert abs(res["e_value"] - want) < 1e-9 * max(1.0, want)
    # P = 1 - exp(-E)
    assert abs(res["p_value"] - (1.0 - math.exp(-res["e_value"]))) < 1e-12


def test_a_better_score_gives_a_smaller_e_value():
    long_hit = blastp("ACDEFGHIKL", "ACDEFGHIKL")["e_value"]
    short_hit = blastp("ACDWWWWWWW", "ACDYYYYYYY")["e_value"]
    assert long_hit < short_hit


def test_mismatch_penalty_is_used():
    strict = blastp("ACDEFG", "ACDEFY", match=1.0, mismatch=-5.0)["score"]
    lax = blastp("ACDEFG", "ACDEFY", match=1.0, mismatch=-0.1)["score"]
    assert strict <= lax


def test_validation():
    for call in (lambda: blastp("", "ACDE"),
                 lambda: blastp("ACDE", "")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
