"""Anchored tests for ldprun.ld_prune (PLINK --indep-pairwise scheme)."""

from morie.fn.ldprun import ld_prune


def _toy():
    # SNP0 == SNP1 (r^2 = 1, equal MAF 0.5 -> later index 1 dropped)
    # SNP3 == 2 - SNP2 (r^2 = 1, equal MAF -> 3 dropped)
    # SNP4 uncorrelated with the survivors (hand-computed r^2:
    # 0-2: sxy=2, sxx=syy=4 -> r^2 = 0.25 < 0.5; 0-4 and 2-4: sxy=0).
    s0 = [0, 1, 2, 0, 1, 2]
    s1 = list(s0)
    s2 = [0, 0, 1, 1, 2, 2]
    s3 = [2 - v for v in s2]
    s4 = [0, 1, 0, 1, 0, 1]
    cols = [s0, s1, s2, s3, s4]
    return [[cols[j][i] for j in range(5)] for i in range(6)]


def test_hand_anchor_prune():
    res = ld_prune(_toy(), window=5, step=2, r2_threshold=0.5)
    assert res["keep"] == [0, 2, 4]
    assert res["drop"] == [1, 3]
    assert res["estimate"] == 3.0
    # hand MAFs: p = (0.5, 0.5, 0.5, 0.5, 0.25)
    expect = [0.5, 0.5, 0.5, 0.5, 0.25]
    assert all(abs(m - e) < 1e-12 for m, e in zip(res["maf"], expect))


def test_threshold_one_keeps_all():
    # r^2 must EXCEED the threshold; at threshold 1.0 nothing is pruned
    res = ld_prune(_toy(), window=5, step=2, r2_threshold=1.0)
    assert res["keep"] == [0, 1, 2, 3, 4]


def test_windowing_limits_pairs():
    # window=2, step=1: only adjacent pairs are ever compared, so the
    # (2,3) pair still prunes, and (0,1) still prunes, but a duplicate
    # placed out of window survives.
    s0 = [0, 1, 2, 0, 1, 2]
    s2 = [0, 0, 1, 1, 2, 2]
    sx = [1, 0, 1, 2, 2, 0]  # buffer, low LD with neighbours
    G = [[s0[i], sx[i], s0[i]] for i in range(6)]
    res = ld_prune(G, window=2, step=1, r2_threshold=0.5)
    # pairs examined: (0,1), (1,2) only; s0-sx r^2 < 0.5 both times
    assert res["keep"] == [0, 1, 2]


def test_maf_tiebreak_lower_dropped():
    # SNP0 rarer than SNP1; identical after recoding? Use proportional
    # columns: s1 = s0 doubled at one entry to give higher MAF while
    # keeping |r| = 1 is impossible with integers; instead use exact
    # copy with different missingness pattern to lower MAF of SNP0.
    s0 = [0, 1, 2, 0, 1, 2, 0, 0]
    s1 = [0, 1, 2, 0, 1, 2, 0, 0]
    s0[7] = 0  # p0 = 6/16 = 0.375
    s1[7] = 2  # p1 = 8/16 = 0.5 -> maf1 = 0.5 > maf0
    # hand-computed r^2 over the 8 pairs: sxy = 4, sxx = 5.5, syy = 6
    # -> r^2 = 16/33 = 0.4848..., so threshold 0.4 trips the pair
    G = [[s0[i], s1[i]] for i in range(8)]
    res = ld_prune(G, window=2, step=1, r2_threshold=0.4)
    assert res["drop"] == [0]  # lower-MAF member dropped
    assert res["keep"] == [1]
