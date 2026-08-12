"""Gabriel haplotype blocks (Gabriel et al. 2002)."""

import math

from ._richresult import RichResult

__all__ = ["hapblk", "haplotype_blocks"]


def _dprime_ci(h, grid=200):
    # 2x2 haplotype counts h = [n00, n01, n10, n11]; likelihood-based
    # one-sided CI on |D'| by profiling the multinomial likelihood
    # over D' with allele frequencies fixed at their MLEs (the
    # confidence-bound approach Gabriel et al. rely on, ref. their
    # (20): Wall & Pritchard-style CI).
    n = sum(h)
    if n == 0:
        return 0.0, 0.0, 0.0
    pA = (h[0] + h[1]) / n          # first locus allele 0 frequency
    pB = (h[0] + h[2]) / n          # second locus allele 0 frequency
    if pA in (0.0, 1.0) or pB in (0.0, 1.0):
        return 0.0, 0.0, 0.0
    p00 = h[0] / n
    D = p00 - pA * pB
    dmax = min(pA * (1 - pB), (1 - pA) * pB) if D > 0 else \
        min(pA * pB, (1 - pA) * (1 - pB))
    if dmax <= 0:
        return 0.0, 0.0, 0.0
    dprime = abs(D) / dmax
    sgn = 1.0 if D >= 0 else -1.0
    # profile likelihood over |D'| in [0, 1]
    logl = []
    for g in range(grid + 1):
        dp = g / grid
        Dg = sgn * dp * dmax
        p = [pA * pB + Dg, pA * (1 - pB) - Dg,
             (1 - pA) * pB - Dg, (1 - pA) * (1 - pB) + Dg]
        if any(v < -1e-12 for v in p):
            logl.append(-1e18)
            continue
        ll = sum(h[k] * math.log(max(p[k], 1e-12)) for k in range(4))
        logl.append(ll)
    tot = max(logl)
    w = [math.exp(v - tot) for v in logl]
    s = sum(w)
    cdf = 0.0
    lo, hi = 0.0, 1.0
    got_lo = False
    for g in range(grid + 1):
        cdf += w[g] / s
        if not got_lo and cdf >= 0.05:
            lo = g / grid
            got_lo = True
        if cdf >= 0.95:
            hi = g / grid
            break
    return dprime, lo, hi


def hapblk(H, strong_hi=0.98, strong_lo=0.70, recomb_hi=0.90,
           frac=0.95):
    """
    Haplotype blocks by the Gabriel et al. (2002) confidence rule.

    As printed in the paper: pairwise LD is measured by D' with
    CONFIDENCE BOUNDS rather than point estimates (D' fluctuates
    upward in small samples); a pair is in "strong LD" when the
    one-sided upper 95% bound exceeds 0.98 (consistent with no
    historical recombination) AND the lower bound is above 0.7; a
    pair shows "strong evidence for historical recombination" when
    the upper bound is below 0.9; other pairs are uninformative.  A
    block is a maximal contiguous marker span in which at least 95%
    of informative pairs are strong-LD.  Bounds come from the
    profile likelihood of |D'| on the 2x2 haplotype table.

    Sources
    -------
    Gabriel, S. B. et al. (2002). The structure of haplotype blocks
    in the human genome. *Science*, 296(5576), 2225-2229, the
    strong-LD/recombination definitions on p. 2226 (local copy
    fetched-wave3/The structure of haplotype blocks in the human
    genome.pdf).

    Parameters
    ----------
    H : matrix (n haplotypes x m markers)
        Binary haplotype data (0/1 alleles).
    strong_hi, strong_lo, recomb_hi : float
        The paper's thresholds (0.98, 0.70, 0.90).
    frac : float
        Required fraction of informative pairs in strong LD (0.95).

    Returns
    -------
    RichResult
        Keys: blocks (list of (start, end) inclusive 0-based),
        dprime (matrix), ci_lo, ci_hi, pair_class ('S'/'R'/'U').
    """
    Hv = [[int(v) for v in row] for row in H]
    n = len(Hv)
    m = len(Hv[0])
    if n < 4 or any(len(r) != m for r in Hv):
        raise ValueError("need >= 4 haplotypes, rectangular")
    dp = [[0.0] * m for _ in range(m)]
    lo_m = [[0.0] * m for _ in range(m)]
    hi_m = [[0.0] * m for _ in range(m)]
    cls = [[""] * m for _ in range(m)]
    for a in range(m):
        for b in range(a + 1, m):
            h = [0, 0, 0, 0]
            for r in Hv:
                h[2 * r[a] + r[b]] += 1
            d, lo, hi = _dprime_ci(h)
            dp[a][b] = dp[b][a] = d
            lo_m[a][b] = lo_m[b][a] = lo
            hi_m[a][b] = hi_m[b][a] = hi
            if hi > strong_hi and lo > strong_lo:
                c = "S"
            elif hi < recomb_hi:
                c = "R"
            else:
                c = "U"
            cls[a][b] = cls[b][a] = c
    blocks = []
    start = 0
    while start < m - 1:
        best_end = -1
        for end in range(m - 1, start, -1):
            ns = nr = 0
            for a in range(start, end + 1):
                for b in range(a + 1, end + 1):
                    if cls[a][b] == "S":
                        ns += 1
                    elif cls[a][b] == "R":
                        nr += 1
            inf = ns + nr
            if inf > 0 and ns / inf >= frac:
                best_end = end
                break
        if best_end > start:
            blocks.append((start, best_end))
            start = best_end + 1
        else:
            start += 1
    return RichResult(payload={
        "blocks": blocks,
        "dprime": dp,
        "ci_lo": lo_m,
        "ci_hi": hi_m,
        "pair_class": cls,
        "method": "Gabriel et al. (2002) confidence-bound blocks",
    })


# long descriptive alias (stub-era name)
haplotype_blocks = hapblk


def cheatsheet():
    return "hapblk: strong LD = CI(D') in (0.7, ...)+hi>0.98; block = 95% strong pairs"
