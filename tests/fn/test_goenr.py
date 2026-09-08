"""Anchored tests for goenr (Boyle et al. 2004 hypergeometric enrichment).

Anchor: exact fraction arithmetic. For N = 10, M = 4, n = 5, k = 3:
P(X >= 3) = [C(4,3) C(6,2) + C(4,4) C(6,1)] / C(10,5)
          = (4 * 15 + 1 * 6) / 252 = 66 / 252 = 11 / 42,
computed by hand from the formula on p. 3711 of the paper.
"""

import math

from morie.fn.goenr import goenr, go_enrichment


def test_goenr_hand_fraction():
    res = goenr(3, 5, 4, 10)
    assert abs(res["pvalue"][0] - 11.0 / 42.0) < 1e-12
    assert abs(res["expected"][0] - 5.0 * 4.0 / 10.0) < 1e-12
    assert abs(res["fold_enrichment"][0] - (3.0 / 5.0) / (4.0 / 10.0)) < 1e-12


def test_goenr_edge_tails():
    # k = 0 -> p = 1 always; k = min(n, M) at the extreme still <= 1
    assert goenr(0, 5, 4, 10)["pvalue"][0] == 1.0
    p_all = goenr(4, 5, 4, 10)["pvalue"][0]
    # P(X = 4) = C(4,4) C(6,1) / C(10,5) = 6/252
    assert abs(p_all - 6.0 / 252.0) < 1e-12


def test_goenr_complement_consistency():
    # upper tail + strict lower tail = 1 (independent identity check)
    N, M, n = 60, 13, 17
    for k in range(0, min(n, M) + 1):
        up = goenr(k, n, M, N)["pvalue"][0]
        # brute-force lower tail from exact binomials
        lo = sum(math.comb(M, i) * math.comb(N - M, n - i)
                 for i in range(0, k)) / math.comb(N, n)
        assert abs(up + lo - 1.0) < 1e-10


def test_goenr_bonferroni_and_vector():
    res = goenr([3, 1], 5, [4, 2], 10, correction="bonferroni")
    assert abs(res["padj"][0] - min(1.0, res["pvalue"][0] * 2)) < 1e-15
    assert res["padj"][1] <= 1.0
    assert go_enrichment is goenr
