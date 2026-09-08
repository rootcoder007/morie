"""Tests for blstn (Altschul et al. 1990, BLAST)."""

import math

from morie.fn.blstn import (blast, blast_pvalue, blstn, estimate_gumbel,
                            karlin_altschul, msp_exact, word_hits)

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
    # lambda and K default to the Karlin-Altschul closed forms, so a
    # p-value is reported without being supplied.
    auto = blstn(target, db, w=8)
    assert auto["hsps"][0]["pvalue"] < 1e-3
    assert auto["lam"] > 0 and auto["K"] > 0


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


def test_karlin_altschul_lambda_is_log3_for_plus_one_minus_one():
    # For +1/-1 scores on uniform DNA, sum p_i e^{lambda s_i} = 1 becomes
    # (1/4)e^{lambda} + (3/4)e^{-lambda} = 1, whose positive root is log 3.
    ka = karlin_altschul(match=1, mismatch=-1)
    assert abs(ka["lam"] - math.log(3.0)) < 1e-10
    assert abs(sum(p * math.exp(ka["lam"] * s)
                   for s, p in ka["distribution"].items()) - 1.0) < 1e-12


def test_karlin_altschul_requirements_and_bounds():
    ka = karlin_altschul(match=5, mismatch=-4)
    assert ka["lam"] > 0 and 0 < ka["K"] < 1
    # K- <= K* <= K+, and the module reports the conservative upper bound.
    assert ka["K_lower"] <= ka["K_upper"]
    assert ka["K"] == ka["K_upper"]
    assert ka["terms"] < 1000        # the series converged inside the cap
    # the bracket is exactly a factor exp(lam * delta) wide
    assert abs(ka["K_upper"] / ka["K_lower"] -
               math.exp(ka["lam"] * ka["delta"])) < 1e-9
    # The mean score must be negative and some score positive, else the
    # equation has no positive root.
    for call in (lambda: karlin_altschul(match=1, mismatch=-0.01),
                 lambda: karlin_altschul(match=-1, mismatch=-2)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_karlin_altschul_bounds_meet_when_the_span_is_small():
    # K- and K+ differ by the factor e^{lambda d}; a finer score lattice
    # squeezes them together.
    # the bracket width is exp(lam * delta); +5/-4 has the gentler lambda
    coarse = karlin_altschul(dist={2: 0.25, -1: 0.75})
    fine = karlin_altschul(match=5, mismatch=-4)
    assert coarse["delta"] == fine["delta"] == 1
    assert fine["lam"] < coarse["lam"]
    assert (fine["K_upper"] / fine["K_lower"] <
            coarse["K_upper"] / coarse["K_lower"])


def test_non_integer_scores_are_rejected_not_truncated():
    for call in (lambda: karlin_altschul(dist={0.05: 0.25, -0.02: 0.75}),
                 lambda: karlin_altschul(match=1, mismatch=-0.01)):
        try:
            call()
            raise AssertionError('expected ValueError')
        except ValueError as e:
            assert 'integer lattice' in str(e)
