# morie.fn -- function file (rootcoder007/morie)
"""Extended haplotype homozygosity (EHH) decay."""

from ._richresult import RichResult

__all__ = ["ehh_decay"]


def _groups_hh(rows):
    """Sum_k n_k (n_k - 1) over identical-row groups, and N (N - 1)."""
    counts = {}
    for r in rows:
        counts[r] = counts.get(r, 0) + 1
    num = sum(c * (c - 1) for c in counts.values())
    n = len(rows)
    return num, n * (n - 1)


def _ehh_curve(hap, core, carriers):
    """EHH at every marker for the chromosome subset ``carriers``.

    At marker j the grouping interval is [min(j, core), max(j, core)]
    inclusive, and EHH = sum_k C(n_k, 2) / C(n, 2) over groups of
    identical haplotypes in that interval (Sabeti et al. 2002; printed
    as displayed equation in Sabeti et al. 2007, Methods p. 6, and as
    eq. 3.1 of the rehh vignette in the n(n-1) form).
    """
    L = len(hap[0])
    n = len(carriers)
    denom = n * (n - 1)
    out = []
    for j in range(L):
        lo, hi = (j, core) if j < core else (core, j)
        rows = [tuple(hap[i][lo:hi + 1]) for i in carriers]
        num, _ = _groups_hh(rows)
        out.append(num / denom)
    return out


def ehh_decay(hap, core, positions=None):
    """EHH decay away from a core SNP, per core allele and site-wise.

    For the chromosomes carrying core allele a (n_a of them), EHH at
    marker x is the probability that two distinct chromosomes drawn
    from that set are identical at every SNP between the core and x,
    inclusive:

        EHH_a(x) = sum_k C(n_k, 2) / C(n_a, 2),

    the n_k being the sizes of the identical-haplotype groups over the
    interval (Sabeti et al. 2002, as printed in Sabeti et al. 2007
    Methods; rehh vignette eq. 3.1).  The site-wise version over ALL N
    chromosomes regardless of core allele (the EHH of Sabeti et al.
    2007 used by XP-EHH; EHHS) is also returned:

        EHHS(x) = sum_i C(n_i, 2) / C(N, 2).

    Both curves equal 1 at the core when every chromosome carrying the
    allele is identical there and decay monotonically outward.

    Parameters
    ----------
    hap : (N, L) array-like of 0/1
        Phased haplotype matrix, chromosomes by SNPs.
    core : int
        Core SNP index (0-based).
    positions : (L,) array-like, optional
        Marker positions (any monotone scale); defaults to the index.

    Returns
    -------
    RichResult
        Keys ``estimate`` (EHH curve of the derived/1 allele),
        ``ehh1``, ``ehh0`` (ancestral/0 allele), ``ehhs`` (all
        chromosomes), ``positions``, ``core``, ``n1``, ``n0``, ``n``,
        ``method``.

    References
    ----------
    Sabeti, P. C., Reich, D. E., et al. (2002). Detecting recent
    positive selection in the human genome from haplotype structure.
    Nature 419(6909), 832-837 (EHH definition; formula read from the
    two sources below, the 2002 PDF being paywalled).
    Sabeti, P. C., Varilly, P., et al. (2007). Genome-wide detection
    and characterization of positive selection in human populations.
    Nature 449(7164), 913-918, Methods p. 6: EHH = sum_i C(n_i, 2) /
    C(N, 2) (fetched-wave3 PDF Sabeti-2007).
    Gautier, M. and Vitalis, R., rehh package vignette sec. 3.1.1
    eq. 3.1 (CRAN; fetched 2026-08-09).
    """
    H = [[int(v) for v in row] for row in hap]
    N = len(H)
    if N < 2:
        raise ValueError("need at least 2 chromosomes")
    L = len(H[0])
    core = int(core)
    if not (0 <= core < L):
        raise ValueError("core out of range")
    for row in H:
        if len(row) != L:
            raise ValueError("ragged haplotype matrix")
        for v in row:
            if v not in (0, 1):
                raise ValueError("haplotypes must be coded 0/1")
    if positions is None:
        pos = [float(j) for j in range(L)]
    else:
        pos = [float(v) for v in positions]
        if len(pos) != L:
            raise ValueError("positions length mismatch")
    car1 = [i for i in range(N) if H[i][core] == 1]
    car0 = [i for i in range(N) if H[i][core] == 0]
    ehh1 = _ehh_curve(H, core, car1) if len(car1) >= 2 else [float("nan")] * L
    ehh0 = _ehh_curve(H, core, car0) if len(car0) >= 2 else [float("nan")] * L
    ehhs = _ehh_curve(H, core, list(range(N)))
    return RichResult(payload={
        "estimate": ehh1, "ehh1": ehh1, "ehh0": ehh0, "ehhs": ehhs,
        "positions": pos, "core": core,
        "n1": len(car1), "n0": len(car0), "n": N,
        "method": "EHH decay (Sabeti 2002/2007), allele-wise and site-wise",
    })


def cheatsheet():
    return "ehhdec: EHH_a(x) = sum C(n_k,2)/C(n_a,2) over [core..x]; plus site-wise EHHS."


# compact alias per ledger/NAMING.md
ehhdecay = ehh_decay
ehhdec = ehh_decay
