# morie.fn -- function file (rootcoder007/morie)
"""LD-based SNP pruning (PLINK --indep-pairwise)."""

from ._richresult import RichResult

__all__ = ["ld_prune"]


def _r2_geno(x, y):
    """Squared Pearson correlation of genotype counts over complete pairs."""
    pairs = [(a, b) for a, b in zip(x, y)
             if a in (0.0, 1.0, 2.0) and b in (0.0, 1.0, 2.0)]
    n = len(pairs)
    if n < 2:
        return float("nan")
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    sxy = sum((a - mx) * (b - my) for a, b in pairs)
    sxx = sum((a - mx) ** 2 for a, _ in pairs)
    syy = sum((b - my) ** 2 for _, b in pairs)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return (sxy * sxy) / (sxx * syy)


def _maf(x):
    obs = [v for v in x if v in (0.0, 1.0, 2.0)]
    if not obs:
        return 0.0
    p = sum(obs) / (2.0 * len(obs))
    return min(p, 1.0 - p)


def ld_prune(G, window=50, step=5, r2_threshold=0.5):
    """LD-based variant pruning, the PLINK ``--indep-pairwise`` scheme.

    A window of ``window`` variants slides along the variant index in
    increments of ``step``.  Within the current window, the squared
    Pearson correlation r^2 between genotype allele counts is computed
    for every pair of still-kept variants ("based on correlations
    between genotype allele counts; phase is not considered"), and
    "pairs of variants in the current window with squared correlation
    greater than the threshold are noted, and variants are greedily
    pruned from the window until no such pairs remain" (PLINK 1.9 LD
    documentation).

    The published description leaves the choice of which pair member
    to drop unspecified, so this implementation pins a deterministic
    rule, documented here and identical in both language arms: offending
    pairs are scanned in row-major index order (i ascending, then j),
    and of the first offending pair the member with the LOWER minor
    allele frequency is removed (the later variant on ties).  Removal
    is permanent across subsequent windows.

    Parameters
    ----------
    G : (n, m) array-like
        Genotype matrix, individuals by variants, coded 0/1/2; any
        other value is treated as missing (pairwise deletion within a
        pair, per-variant deletion for MAF).
    window : int
        Window size in variant count.
    step : int
        Window shift in variant count.
    r2_threshold : float
        Pairwise r^2 above which one member of a pair is pruned.

    Returns
    -------
    RichResult
        Keys ``keep`` (kept variant indices, the prune.in set),
        ``drop`` (removed indices, prune.out), ``estimate`` (number
        kept), ``maf``, ``n_variants``, ``method``.

    References
    ----------
    Purcell, S., Neale, B., et al. (2007). PLINK: a tool set for
    whole-genome association and population-based linkage analyses.
    American Journal of Human Genetics 81(3), 559-575 (sec. "Linkage
    disequilibrium based SNP pruning").
    PLINK 1.9 LD documentation, --indep-pairwise,
    https://www.cog-genomics.org/plink/1.9/ld (algorithm text quoted
    above; fetched 2026-08-09).
    """
    rows = [[float(v) for v in row] for row in G]
    n = len(rows)
    if n == 0:
        raise ValueError("empty genotype matrix")
    m = len(rows[0])
    cols = [[rows[i][j] for i in range(n)] for j in range(m)]
    window = int(window)
    step = int(step)
    if window < 2 or step < 1:
        raise ValueError("need window >= 2 and step >= 1")
    if not (0.0 < float(r2_threshold) <= 1.0):
        raise ValueError("r2_threshold must be in (0, 1]")
    mafs = [_maf(c) for c in cols]
    removed = [False] * m
    start = 0
    while True:
        end = min(start + window, m)
        active = [j for j in range(start, end) if not removed[j]]
        while True:
            offender = None
            for ai in range(len(active)):
                for bi in range(ai + 1, len(active)):
                    i, j = active[ai], active[bi]
                    r2 = _r2_geno(cols[i], cols[j])
                    if r2 == r2 and r2 > float(r2_threshold):
                        offender = (i, j)
                        break
                if offender is not None:
                    break
            if offender is None:
                break
            i, j = offender
            drop = j if mafs[j] <= mafs[i] else i
            removed[drop] = True
            active = [k for k in active if k != drop]
        if end >= m:
            break
        start += step
    keep = [j for j in range(m) if not removed[j]]
    dropped = [j for j in range(m) if removed[j]]
    return RichResult(payload={
        "estimate": float(len(keep)),
        "keep": keep, "drop": dropped, "maf": mafs,
        "n_variants": int(m),
        "method": "LD pruning, PLINK --indep-pairwise (lower-MAF member dropped)",
    })


def cheatsheet():
    return "ldprun: PLINK --indep-pairwise LD pruning; drops lower-MAF member of r2>thr pairs."


# compact alias per ledger/NAMING.md; ld_pruning/ldpruning are the
# pre-existing exported names, kept as aliases
ldprune = ld_prune
ldprun = ld_prune
ld_pruning = ld_prune
ldpruning = ld_prune
