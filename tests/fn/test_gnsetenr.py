"""Anchored tests for gnsetenr (Subramanian et al. 2005 GSEA ES).

Hand anchor (Appendix, Enrichment Score ES(S)): ranked r = (3, 2, 1,
-1, -2), set = {rank 1, rank 3}, p = 1. N_R = 3 + 1 = 4, miss step
1/3. Running sum: 3/4, 3/4 - 1/3 = 5/12, 5/12 + 1/4 = 2/3, 1/3, 0.
ES = 3/4 at rank 1 (0-based 0).
"""

from morie.fn.gnsetenr import gnsetenr, geneset_enrichment


def test_gnsetenr_hand_running_sum():
    r = [3.0, 2.0, 1.0, -1.0, -2.0]
    mem = [1, 0, 1, 0, 0]
    res = gnsetenr(r, mem, p=1.0)
    assert abs(res["es"] - 0.75) < 1e-15
    assert res["arg_es"] == 0
    expect = [3.0 / 4.0, 3.0 / 4.0 - 1.0 / 3.0,
              3.0 / 4.0 - 1.0 / 3.0 + 1.0 / 4.0, 1.0 / 3.0, 0.0]
    for a, b in zip(res["running"], expect):
        assert abs(a - b) < 1e-12
    assert res["n_hits"] == 2


def test_gnsetenr_p0_is_ks():
    # p = 0: ES is the classical two-sample KS statistic between the
    # ranks of hits and misses (paper, p. 15550). Independent route:
    # direct max_i |H(i)/NH - M(i)/(N-NH)|.
    r = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, -1.0, -3.0]
    mem = [0, 1, 1, 0, 0, 1, 0, 0]
    res = gnsetenr(r, mem, p=0.0)
    nh = sum(mem)
    n = len(r)
    h = m = 0
    best = 0.0
    for i in range(n):
        if mem[i]:
            h += 1
        else:
            m += 1
        dev = h / nh - m / (n - nh)
        if abs(dev) > abs(best):
            best = dev
    assert abs(res["es"] - best) < 1e-12


def test_gnsetenr_unordered_input_sorted_internally():
    r = [1.0, 3.0, -2.0, 2.0, -1.0]
    mem = [1, 1, 0, 0, 0]
    r2 = [3.0, 2.0, 1.0, -1.0, -2.0]
    mem2 = [1, 0, 1, 0, 0]
    a = gnsetenr(r, mem)
    b = gnsetenr(r2, mem2)
    assert abs(a["es"] - b["es"]) < 1e-15


def test_gnsetenr_permutation_pvalue_deterministic():
    r = [3.0, 2.5, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -2.5, -3.0]
    mem = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    a = gnsetenr(r, mem, nperm=200, seed=42)
    b = gnsetenr(r, mem, nperm=200, seed=42)
    assert a["pvalue"] == b["pvalue"]
    assert 0.0 <= a["pvalue"] <= 1.0
    assert geneset_enrichment is gnsetenr
