import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_uniform
from morie.fn.pppextra import (
    cross_k,
    knn_distances,
    morisita_index,
    potts_exact,
    potts_gibbs,
    quadrat_test,
    segregation_test,
)

U = [float(v) for v in random_uniform(120, seed=23, stream=0)]
PTS = [(U[i], U[40 + i]) for i in range(40)]
MK = ["a" if U[80 + i] < 0.5 else "b" for i in range(40)]


def test_knn_distances_bruteforce():
    r = knn_distances(PTS, k=3)
    for i, p in enumerate(PTS):
        d = sorted(math.dist(p, q) for j, q in enumerate(PTS) if j != i)
        assert r.distance[i] == d[2]
    assert knn_distances([(0, 0), (1, 0), (3, 0)], k=2).distance == [3.0, 2.0, 3.0]


def test_quadrat_and_morisita_formulas():
    r = quadrat_test(PTS, (0, 1, 0, 1), 3, 2)
    E = 40 / 6
    assert abs(r.statistic - ssum((c - E) ** 2 / E for row in r.counts for c in row)) < 1e-12
    assert sum(map(sum, r.counts)) == 40 and r.df == 5 and abs(r.intensity - 40) < 1e-12
    cnt = [3, 0, 5, 1, 1, 9, 0, 2]
    m = morisita_index(cnt)
    s, s2 = 21, sum(v * v for v in cnt)
    assert abs(m.imor - 8 * (s2 - s) / (s * s - s)) < 1e-15
    assert m.imor > m.mclu > 1 and 0.5 <= m.imst <= 1


def test_cross_k_interior_pairs_unweighted():
    r = cross_k(PTS, MK, "a", "b", (0, 1, 0, 1), [0.05, 0.1])
    pa = [p for p, m in zip(PTS, MK) if m == "a"]
    pb = [p for p, m in zip(PTS, MK) if m == "b"]
    for rr, K in zip(r.r, r.K):
        assert sum(1 for a in pa for b in pb if math.dist(a, b) <= rr) / (len(pa) * len(pb)) <= K
    assert abs(r.L[1] - math.sqrt(r.K[1] / math.pi)) < 1e-15


def test_segregation_statistic_and_pvalue():
    r = segregation_test(PTS, MK, 0.1, nsim=9, seed=3)
    n = 40
    K = [[math.exp(-(math.dist(PTS[a], PTS[b]) ** 2) / 0.02) if a != b else 0 for b in range(n)] for a in range(n)]
    T = 0.0
    for m in ("a", "b"):
        pbar = MK.count(m) / n
        for a in range(n):
            T += (ssum(K[a][b] for b in range(n) if MK[b] == m) / ssum(K[a]) - pbar) ** 2
    assert abs(r.statistic - T) < 1e-12
    assert r.p_value == (1 + sum(1 for v in r.simulated if v >= r.statistic)) / 10


def test_potts_gibbs_matches_exact_expectation():
    ex = potts_exact(3, 3, 2, 0.6)
    g = potts_gibbs(3, 3, 2, 0.6, sweeps=6000, seed=11)
    s = g.like_pairs[200:]
    b = 20
    size = len(s) // b
    means = [ssum(s[k * size : (k + 1) * size]) / size for k in range(b)]
    mu = ssum(means) / b
    se = math.sqrt(ssum((v - mu) ** 2 for v in means) / (b - 1) / b)
    assert abs(mu - ex.mean_like_pairs) / se < 4
    assert potts_gibbs(3, 3, 2, 50.0, sweeps=5, init=[[0] * 3] * 3).like_pairs[-1] == 12
    assert abs(math.exp(potts_exact(1, 2, 2, 0.0).log_z) - 4) < 1e-12
