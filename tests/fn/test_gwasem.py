"""Tests for gwasem (Kang et al. 2010, EMMAX)."""

import math

from morie.fn.gwasem import (emmax, genomic_control, gower_normalize, gwasem,
                             kinship_ibs, reml_variance)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _panel(n=60, m=150, seed=101, effect=0.0):
    rnd = _lcg(seed)

    def z():
        return math.sqrt(-2 * math.log(max(rnd(), 1e-12))) * \
            math.cos(2 * math.pi * rnd())

    pop = [0 if i < n // 2 else 1 for i in range(n)]
    G = [[sum(1 for _ in range(2)
              if rnd() < (0.2 if pop[i] == 0 else 0.6))
          for _ in range(m)] for i in range(n)]
    y = [2.0 * pop[i] + effect * G[i][0] + z() for i in range(n)]
    return y, G, pop


def test_gower_normalisation_is_an_identity():
    raw = [[2.0, 0.4, 0.1], [0.4, 1.6, 0.3], [0.1, 0.3, 2.2]]
    SN = gower_normalize(raw)
    n = 3
    rows = [sum(r) / n for r in SN]
    tot = sum(rows) / n
    tr = sum(SN[i][i] - 2.0 * rows[i] + tot for i in range(n))
    assert abs(tr - (n - 1.0)) < 1e-9
    try:
        gower_normalize([[1.0, 1.0], [1.0, 1.0]])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_ibs_kinship():
    S = kinship_ibs([[0, 0, 2, 2], [0, 0, 2, 2], [2, 2, 0, 0]])
    assert S[0][1] == 1.0 and S[0][2] == 0.0 and S[2][2] == 1.0


def test_identity_kinship_reduces_to_ols():
    n = 40
    rnd = _lcg(31)
    x = [1.0 if rnd() < 0.5 else 0.0 for _ in range(n)]
    y = [1.5 + 2.0 * x[i] + 0.5 *
         (math.sqrt(-2 * math.log(max(rnd(), 1e-12))) *
          math.cos(2 * math.pi * rnd())) for i in range(n)]
    identity = [[1.0 if i == k else 0.0 for k in range(n)]
                for i in range(n)]
    res = gwasem(y, [[v] for v in x], kinship=identity)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    b = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / sxx
    a = my - b * mx
    rss = sum((y[i] - a - b * x[i]) ** 2 for i in range(n))
    se = math.sqrt(rss / (n - 2) / sxx)
    assert abs(res["beta"][0] - b) < 1e-8
    assert abs(res["se"][0] - se) < 1e-8
    assert abs(res["stat"][0] - (b / se) ** 2) < 1e-6


def test_emmax_controls_inflation_where_uncorrected_does_not():
    y, G, _ = _panel()
    n = len(y)
    em = gwasem(y, G)
    identity = [[1.0 if i == k else 0.0 for k in range(n)]
                for i in range(n)]
    un = gwasem(y, G, kinship=identity)
    assert un["lambda_gc"] > 5.0
    assert em["lambda_gc"] < 2.5
    assert un["lambda_gc"] > 3.0 * em["lambda_gc"]
    assert em["pseudo_heritability"] > 0.3


def test_recovers_a_planted_effect():
    # n = 100: at n = 60 the effect is still recovered as the top hit, but
    # the point estimate carries enough noise that pinning it is a test of
    # the sample rather than of the estimator
    y, G, _ = _panel(n=100, effect=1.2)
    r = gwasem(y, G)
    assert abs(r["beta"][0] - 1.2) < 0.5
    assert r["pvalue"][0] == min(r["pvalue"])
    sc = gwasem(y, G, test="score")
    assert abs(sc["beta"][0] - r["beta"][0]) < 1e-9


def test_reml_reports_a_variance_decomposition():
    y, G, _ = _panel()
    vc = reml_variance(y, kinship_ibs(G))
    assert vc["sigma_a2"] > 0 and vc["sigma_e2"] > 0
    assert 0.0 <= vc["pseudo_heritability"] <= 1.0
    assert abs(vc["pseudo_heritability"] -
               vc["sigma_a2"] / (vc["sigma_a2"] + vc["sigma_e2"])) < 1e-12
    assert vc["lrt"] >= 0.0


def test_genomic_control():
    # a null chi-square sample sits at 1
    rnd = _lcg(9)
    chis = []
    for _ in range(400):
        z = math.sqrt(-2 * math.log(max(rnd(), 1e-12))) * \
            math.cos(2 * math.pi * rnd())
        chis.append(z * z)
    assert abs(genomic_control(chis) - 1.0) < 0.3
    assert genomic_control([v * 4 for v in chis]) > 3.0


def test_validation():
    y, G, _ = _panel(n=20, m=30)
    for call in (lambda: gwasem(y[:-1], G),
                 lambda: gwasem([1.0, 2.0], [[1, 2], [1]]),
                 lambda: gwasem(y, G, trait="ordinal"),
                 lambda: gwasem(y, G, test="wald"),
                 lambda: gwasem(y, G, trait="binary"),
                 lambda: reml_variance(y, [[1.0, 0.0], [0.0, 1.0]]),
                 lambda: gower_normalize([[1.0]])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert emmax is gwasem
