# morie.fn -- function file (rootcoder007/morie)
r"""Clumping and thresholding: one variant per signal, not one per
correlated neighbour.

A polygenic score is a weighted sum of alleles, and the weights come
from a genome-wide association study in which neighbouring variants are
correlated by descent. Adding all of them counts the same signal as many
times as the chip happens to type it, so the score is dominated by
whichever region was densest on the array rather than whichever region
matters.

Clumping fixes that greedily, and the greed is the method rather than an
approximation to it. Take the most significant variant not yet assigned;
it becomes an index variant. Every variant within ``window`` of it whose
squared correlation exceeds ``r2`` is assigned to its clump and removed.
Repeat on what is left. The result is one representative per correlated
block, chosen by significance, and the choice is deterministic --
ties in the p-value are broken by position and then by index, so two
implementations agree on the clumps and not merely on their number.

Thresholding is then a separate decision: for each p-value cut-off, the
score uses the index variants below it,

.. math:: \mathrm{PRS}_i = \sum_{j\in S(T)} \hat\beta_j\,x_{ij}.

Several thresholds are always computed, because the best one is a
property of the target sample and not of the summary statistics, and
reporting a single number hides that the choice was made. The
degenerate settings are exact: ``r2 = 1`` clumps nothing, ``r2 = 0``
keeps exactly one variant per fully correlated block, and a threshold of
zero gives an identically zero score.

References
----------
Purcell, S., Neale, B., Todd-Brown, K. et al. (2007) "PLINK: a tool set
for whole-genome association and population-based linkage analyses",
*American Journal of Human Genetics* **81**(3), 559-575,
doi:10.1086/519795. The clumping algorithm implemented here.

International Schizophrenia Consortium, Purcell, S. M., Wray, N. R.,
Stone, J. L., Visscher, P. M., O'Donovan, M. C., Sullivan, P. F. and
Sklar, P. (2009) "Common polygenic variation contributes to risk of
schizophrenia and bipolar disorder", *Nature* **460**, 748-752,
doi:10.1038/nature08185. Clumping and thresholding as a score.

Choi, S. W. and O'Reilly, P. F. (2019) "PRSice-2: Polygenic Risk Score
software for biobank-scale data", *GigaScience* **8**(7), giz082,
doi:10.1093/gigascience/giz082.

Choi, S. W., Mak, T. S.-H. and O'Reilly, P. F. (2020) "Tutorial: a
guide to performing polygenic risk score analyses", *Nature Protocols*
**15**(9), 2759-2772, doi:10.1038/s41596-020-0353-1.

Prive, F., Vilhjalmsson, B. J., Aschard, H. and Blum, M. G. B. (2019)
"Making the most of clumping and thresholding for polygenic scores",
*American Journal of Human Genetics* **105**(6), 1213-1221,
doi:10.1016/j.ajhg.2019.11.001.

Wray, N. R., Yang, J., Hayes, B. J., Price, A. L., Goddard, M. E. and
Visscher, P. M. (2013) "Pitfalls of predicting complex traits from
SNPs", *Nature Reviews Genetics* **14**(7), 507-515,
doi:10.1038/nrg3457.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["prs_cs_clump"]

_EPS = 1e-12
_DEFAULT_THRESHOLDS = [5e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]


def prs_cs_clump(sumstats, ld_ref, p_threshold=None, r2=0.1, window=250000.0,
                 genotypes=None, standardize=False):
    r"""Clump correlated variants, then score at one or more thresholds.

    Parameters
    ----------
    sumstats : mapping
        ``beta`` and ``p`` (each length ``m``), optionally ``position``
        and ``snp``. Position defaults to the column index, which makes
        ``window`` a count of variants rather than base pairs -- fine
        for a single block, wrong for a genome, so pass positions.
    ld_ref : array-like, shape ``(m, m)``
        Squared correlations between variants. Must be symmetric with a
        unit diagonal; correlations rather than squared correlations are
        detected by a negative entry and rejected, because silently
        squaring them would change the answer without saying so.
    p_threshold : float or sequence, optional
        Thresholds to score at. Defaults to the nine conventional ones.
    r2 : float
        Clumping threshold. ``1`` clumps nothing.
    window : float
        Maximum distance at which two variants can be clumped.
    genotypes : array-like, shape ``(n, m)``, optional
        Target-sample dosages. Without them the retained variants and
        weights are returned but no score is computed.

    Returns
    -------
    RichResult
        ``index_variants`` and ``clump_of`` per variant, the retained
        set and score at each threshold, and the per-individual
        ``score`` at the smallest threshold that retains anything.
    """
    if not isinstance(sumstats, dict):
        raise ValueError("prsclm: sumstats must be a mapping with 'beta' "
                         "and 'p'")
    for key in ("beta", "p"):
        if key not in sumstats:
            raise ValueError("prsclm: sumstats is missing '%s'" % key)
    beta = [float(v) for v in k.vec(sumstats["beta"])]
    pv = [float(v) for v in k.vec(sumstats["p"])]
    m = len(beta)
    if m == 0:
        raise ValueError("prsclm: no variants")
    if len(pv) != m:
        raise ValueError("prsclm: %d effect sizes but %d p-values"
                         % (m, len(pv)))
    if any(v < 0.0 or v > 1.0 for v in pv):
        raise ValueError("prsclm: a p-value outside [0, 1]")
    pos = ([float(i) for i in range(m)] if "position" not in sumstats
           else [float(v) for v in k.vec(sumstats["position"])])
    if len(pos) != m:
        raise ValueError("prsclm: %d variants but %d positions"
                         % (m, len(pos)))
    names = (["v%d" % i for i in range(m)] if "snp" not in sumstats
             else [str(v) for v in sumstats["snp"]])
    if len(names) != m:
        raise ValueError("prsclm: %d variants but %d names" % (m, len(names)))

    R = [[float(v) for v in row] for row in k.mat(ld_ref)]
    if len(R) != m or any(len(r) != m for r in R):
        raise ValueError("prsclm: ld_ref must be %d by %d" % (m, m))
    if any(R[i][j] < -1e-9 for i in range(m) for j in range(m)):
        raise ValueError("prsclm: ld_ref holds a negative entry -- it must "
                         "be squared correlations, not correlations; "
                         "squaring them here would change the answer "
                         "without saying so")
    asym = max(abs(R[i][j] - R[j][i]) for i in range(m) for j in range(m))
    if asym > 1e-8:
        raise ValueError("prsclm: ld_ref is not symmetric (largest "
                         "asymmetry %.3g)" % asym)
    r2t = float(r2)
    if not 0.0 <= r2t <= 1.0:
        raise ValueError("prsclm: r2 must be in [0, 1]")
    win = float(window)
    if win < 0.0:
        raise ValueError("prsclm: the window cannot be negative")

    if p_threshold is None:
        thr = list(_DEFAULT_THRESHOLDS)
    elif isinstance(p_threshold, (int, float)):
        thr = [float(p_threshold)]
    else:
        thr = [float(v) for v in k.vec(p_threshold)]
    if any(v < 0.0 or v > 1.0 for v in thr):
        raise ValueError("prsclm: a threshold outside [0, 1]")
    thr = sorted(set(thr))

    # ---- PLINK clumping: most significant first, ties by position then index
    order = sorted(range(m), key=lambda i: (pv[i], pos[i], i))
    clump_of = [-1] * m
    clumps = []
    for i in order:
        if clump_of[i] != -1:
            continue
        clump_of[i] = i
        grp = [i]
        for j in range(m):
            if clump_of[j] != -1 or j == i:
                continue
            if abs(pos[j] - pos[i]) <= win and R[i][j] > r2t:
                clump_of[j] = i
                grp.append(j)
        clumps.append((i, sorted(grp)))
    clumps.sort(key=lambda c: c[0])
    index_variants = [c[0] for c in clumps]
    members = [c[1] for c in clumps]

    # every index variant must be the most significant in its own clump --
    # if it is not, the greedy order was applied wrongly
    index_is_top = all(pv[i] <= min(pv[j] for j in grp) + 1e-15
                       for i, grp in clumps)

    G = None
    n = 0
    if genotypes is not None:
        G = [[float(v) for v in row] for row in k.mat(genotypes)]
        n = len(G)
        if n and any(len(r) != m for r in G):
            raise ValueError("prsclm: genotypes must have %d columns" % m)
        if standardize:
            for a in range(m):
                col = [G[i][a] for i in range(n)]
                mu = sum(col) / n
                sd = math.sqrt(sum((v - mu) ** 2 for v in col)
                               / max(n - 1, 1))
                if sd > _EPS:
                    for i in range(n):
                        G[i][a] = (G[i][a] - mu) / sd
                else:
                    for i in range(n):
                        G[i][a] = 0.0

    retained, scores, counts = [], [], []
    for t in thr:
        keep = [i for i in index_variants if pv[i] < t]
        retained.append(keep)
        counts.append(len(keep))
        if G is None:
            scores.append(None)
        else:
            scores.append([sum(beta[j] * G[i][j] for j in keep)
                           for i in range(n)])

    best = None
    for u, t in enumerate(thr):
        if counts[u] > 0:
            best = u
            break

    return RichResult(payload={
        "estimate": counts, "n_retained": counts,
        "thresholds": thr, "retained": retained,
        "score": (scores[best] if best is not None and G is not None
                  else None),
        "score_threshold": (thr[best] if best is not None else None),
        "scores_by_threshold": scores,
        "index_variants": [float(v) for v in index_variants],
        "index_variant_names": [names[i] for i in index_variants],
        "clump_of": [float(v) for v in clump_of],
        "clump_members": [[float(v) for v in grp] for grp in members],
        "clump_sizes": [len(grp) for grp in members],
        "index_is_most_significant": index_is_top,
        "n_clumps": len(members), "n_variants": m, "n_individuals": n,
        "r2": r2t, "window": win, "standardized": bool(standardize),
        "weights": beta,
        "method": "PLINK-style LD clumping (most significant index variant "
                  "first, correlated neighbours within the window removed) "
                  "followed by p-value thresholding (Purcell et al. 2007; "
                  "International Schizophrenia Consortium 2009; Choi & "
                  "O'Reilly 2019)",
        "note": "index_variants and clump_of are 0-based; several "
                "thresholds are always scored because the best one is a "
                "property of the target sample, and reporting one number "
                "would hide that a choice was made",
    })


def cheatsheet():
    return ("prsclm: prs_cs_clump(sumstats, ld_ref, p_threshold) -> LD "
            "clumping plus thresholded polygenic scores (Purcell et al. "
            "2007 PLINK; Choi & O'Reilly 2019 PRSice-2)")
