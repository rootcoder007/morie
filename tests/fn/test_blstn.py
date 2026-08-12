"""Tests for blstn (Altschul et al. 1990, BLAST)."""

import math

from morie.fn.blstn import (blast, blast_pvalue, blstn, estimate_gumbel,
                            msp_exact, word_hits)

ALPHA = "ACGT"


def _lcg(seed):
    state = [seed]

    def f():
        state[0] = (1103515245 * state[0] + 12345) % (1 << 31)
        return state[0] / float(1 << 31)
    return f


def _brute_msp(q, s, match=5, mismatch=-4):
    best = 0.0
    for i in range(len(q)):
        for j in range(len(s)):
            for length in range(1, min(len(q) - i, len(s) - j) + 1):
                tot = sum(match if q[i + t] == s[j + t] else mismatch
                          for t in range(length))
                best = max(best, tot)
    return best


def test_msp_exact_matches_brute_force():
    rand = _lcg(31)
    for _ in range(6):
        q = "".join(ALPHA[int(rand() * 4)] for _ in range(14))
        s = "".join(ALPHA[int(rand() * 4)] for _ in range(16))
        assert msp_exact(q, s)[0] == _brute_msp(q, s)


def test_msp_coordinates_carry_the_reported_score():
    q, s = "ACGTACGTTTGACC", "TTTACGTACGTTTGACCGG"
    score, qi, sj, length = msp_exact(q, s)
    assert score == sum(5 if q[qi + t] == s[sj + t] else -4
                        for t in range(length))
    assert msp_exact(q, q)[0] == 5 * len(q)


def test_blast_finds_a_planted_region():
    q = "ACGTACGTTTGACCAGGTAAC"
    s = "TTTTTTACGTACGTTTGACCAGGTAACGGG"
    res = blstn(q, s, w=8)
    assert res["best_score"] == 105.0
    assert res["n_hsps"] == 1
    h = res["hsps"][0]
    assert h["sstart"] == 6 and h["identities"] == h["length"]


def test_the_heuristic_never_beats_the_exact_msp():
    rand = _lcg(99)
    missed = 0
    for _ in range(25):
        a = "".join(ALPHA[int(rand() * 4)] for _ in range(60))
        b = "".join(ALPHA[int(rand() * 4)] for _ in range(60))
        exact = msp_exact(a, b)[0]
        heur = blstn(a, b, w=4, cutoff=0.0)["best_score"]
        assert heur <= exact + 1e-9
        missed += heur < exact - 1e-9
    # and it is a real heuristic: sometimes it misses
    assert missed > 0


def test_an_unbounded_x_drop_reaches_the_exact_msp():
    rand = _lcg(5150)
    a = "".join(ALPHA[int(rand() * 4)] for _ in range(200))
    b = a[:40] + "".join(ALPHA[int(rand() * 4)] for _ in range(60)) + a[100:]
    loose = blstn(a, b, w=8, X=1000, cutoff=0.0)["best_score"]
    assert abs(loose - msp_exact(a, b)[0]) < 1e-9


def test_neighborhood_words_find_inexact_hits():
    mat = [[1.0 if i == j else -1.0 for j in range(4)] for i in range(4)]
    q, s = "ACGTACGT", "ACGAACGT"
    exact = word_hits(q, s, 4, "exact")
    near = word_hits(q, s, 4, "neighborhood", threshold=2.0, matrix=mat,
                     alphabet=ALPHA)
    assert len(near) > len(exact)
    perfect = word_hits(q, s, 4, "neighborhood", threshold=4.0, matrix=mat,
                        alphabet=ALPHA)
    assert sorted(perfect) == sorted(exact)


def test_pvalue_equations():
    lam, K, m, n, S = 0.2, 0.1, 250, 250, 50.0
    y = K * m * n * math.exp(-lam * S)
    assert abs(blast_pvalue(S, m, n, lam, K) - (1 - math.exp(-y))) < 1e-12
    for c in (2, 3):
        tail = sum(y ** i / math.factorial(i) for i in range(c))
        assert abs(blast_pvalue(S, m, n, lam, K, c) -
                   (1 - math.exp(-y) * tail)) < 1e-12
    assert (blast_pvalue(30.0, m, n, lam, K) >
            blast_pvalue(50.0, m, n, lam, K))


def test_estimated_gumbel_predicts_held_out_exceedances():
    fit = estimate_gumbel(80, 80, [0.25] * 4, n_sim=200, seed=11)
    held = estimate_gumbel(80, 80, [0.25] * 4, n_sim=200, seed=777)["scores"]
    assert fit["lam"] > 0 and fit["K"] > 0
    for S in (45.0, 55.0):
        obs = sum(1 for v in held if v >= S) / float(len(held))
        pred = blast_pvalue(S, 80, 80, fit["lam"], fit["K"])
        assert abs(obs - pred) < 0.15


def test_database_search_finds_the_planted_subject():
    rand = _lcg(4242)
    target = "GGCATTACGTGACCTTAGGCAT"
    db = ["".join(ALPHA[int(rand() * 4)] for _ in range(120))
          for _ in range(6)]
    db[3] = db[3][:40] + target + db[3][40 + len(target):]
    out = blstn(target, db, w=8, lam=0.22, K=0.37)
    assert out["hsps"][0]["subject"] == 3
    assert out["hsps"][0]["sstart"] == 40
    assert out["hsps"][0]["pvalue"] < 1e-3
    assert "pvalue" not in blstn(target, db, w=8)["hsps"][0]


def test_validation():
    for call in (lambda: blstn("", "ACGT"),
                 lambda: blstn("ACGT", []),
                 lambda: blstn("ACGT", "ACGT", w=0),
                 lambda: blstn("ACGT", "ACGT", X=-1),
                 lambda: word_hits("ACGT", "ACGT", 2, "neighborhood"),
                 lambda: blast_pvalue(10, 10, 10, 0.0, 0.1)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert blast is blstn
