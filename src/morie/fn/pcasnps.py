# morie.fn -- function file (rootcoder007/morie)
"""PCA on a genotype matrix (Patterson-Price-Reich eigenanalysis)."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["pca_snps"]


def pca_snps(genotypes, n_components=2):
    """Principal components of a SNP matrix, normalized as in EIGENSTRAT.

    The normalization is the whole method.  Centring alone leaves each
    marker weighted by its own allele frequency, so common SNPs
    dominate the leading components for no genetic reason.  Patterson,
    Price & Reich divide instead by the drift standard deviation the
    binomial model predicts,

        p_j = mu_j / 2,
        M_ij = (g_ij - mu_j) / sqrt(p_j (1 - p_j)),

    which puts every marker on the same footing, and then take the
    eigenvectors of ``X = M M' / m`` over INDIVIDUALS -- an ``n by n``
    problem regardless of how many markers there are.

    The significance of the leading eigenvalue is judged against the
    Tracy-Widom law.  With ``l1' = n lambda_1 / sum(lambda)`` the
    normalized top eigenvalue and

        n' = (n + 1) (sum l)^2 / ((n - 1) sum l^2 - (sum l)^2),
        mu = (sqrt(n - 1) + sqrt(n'))^2 / n',
        sigma = ((sqrt(n-1) + sqrt(n')) / n')
                (1/sqrt(n-1) + 1/sqrt(n'))^{1/3},

    the statistic ``(l1' - mu) / sigma`` follows TW1 under the null of
    no structure.  ``n'`` is the effective number of markers, and it is
    NOT ``m``: linkage makes the markers less independent than they
    look, and using ``m`` overstates significance.

    Monomorphic markers carry no information and would divide by zero;
    they are dropped and counted rather than left to produce ``inf``.

    Parameters
    ----------
    genotypes : array-like, shape (n, m)
        Genotype counts in {0, 1, 2}; individuals by markers.
    n_components : int, default 2
        Number of leading components to return.

    Returns
    -------
    RichResult
        ``eigenvalues`` (descending), ``pcs`` (n by n_components),
        ``variance_explained``, ``estimate`` (the leading eigenvalue),
        ``tw_statistic``, ``n_eff``, ``n_dropped``, ``n``, ``m``.

    References
    ----------
    Patterson, N., Price, A. L. & Reich, D. (2006).  Population
    structure and eigenanalysis.  PLoS Genetics, 2(12), e190.
    doi:10.1371/journal.pgen.0020190
    """
    G = [[float(v) for v in row] for row in genotypes]
    n = len(G)
    if n < 2:
        raise ValueError("pca_snps: need at least two individuals")
    m0 = len(G[0])
    if any(len(r) != m0 for r in G):
        raise ValueError("pca_snps: ragged genotype matrix")
    if m0 == 0:
        raise ValueError("pca_snps: no markers")
    k = int(n_components)
    if k < 1 or k > n:
        raise ValueError("pca_snps: n_components must satisfy 1 <= k <= n")

    cols = []
    dropped = 0
    for j in range(m0):
        mu = sum(G[i][j] for i in range(n)) / n
        pj = mu / 2.0
        v = pj * (1.0 - pj)
        if v <= 0.0:
            dropped += 1
            continue
        s = math.sqrt(v)
        cols.append([(G[i][j] - mu) / s for i in range(n)])
    m = len(cols)
    if m == 0:
        raise ValueError("pca_snps: every marker is monomorphic")

    Xc = [[0.0] * n for _ in range(n)]
    for c in cols:
        for i in range(n):
            ci = c[i]
            for j in range(n):
                Xc[i][j] += ci * c[j]
    for i in range(n):
        for j in range(n):
            Xc[i][j] /= m
    vals, vecs = core.jacobi(Xc)
    order = list(range(n - 1, -1, -1))
    ev = [vals[i] for i in order]
    pcs = [[vecs[i][order[t]] for t in range(k)] for i in range(n)]
    tot = sum(ev)
    vexp = [v / tot if tot > 0.0 else float("nan") for v in ev]

    s1 = sum(ev)
    s2 = sum(v * v for v in ev)
    den = (n - 1.0) * s2 - s1 * s1
    neff = (n + 1.0) * s1 * s1 / den if den > 0.0 else float("nan")
    if neff == neff and neff > 0.0:
        a = math.sqrt(n - 1.0) + math.sqrt(neff)
        mu_tw = a * a / neff
        sg = (a / neff) * (1.0 / math.sqrt(n - 1.0) + 1.0 / math.sqrt(neff)) ** (1.0 / 3.0)
        l1 = n * ev[0] / s1 if s1 > 0.0 else float("nan")
        tw = (l1 - mu_tw) / sg
    else:
        tw = float("nan")
    return RichResult(payload={
        "eigenvalues": ev, "pcs": pcs, "variance_explained": vexp,
        "estimate": ev[0], "tw_statistic": tw, "n_eff": neff,
        "n_dropped": float(dropped), "n": n, "m": m,
        "method": "EIGENSTRAT genotype PCA (Patterson-Price-Reich 2006)"})


def cheatsheet():
    return "pcasnps: EIGENSTRAT PCA on a genotype matrix"


pcasnps = pca_snps
