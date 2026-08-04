# morie.fn -- function file (rootcoder007/morie)
"""Identification of a binary-response model under median independence.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 4.2, Theorem 4.1 (page 99).

Model:  Y = I(X'beta + U >= 0),  median(U | X = x) = 0.

Theorem 4.1 -- with |beta_1| = 1, beta is identified if

  (a) the support of X is not contained in any proper linear subspace
      of R^d;
  (b) for almost every xtilde = (x_2, ..., x_d) the distribution of
      X_1 conditional on Xtilde = xtilde has an everywhere positive
      density.

Mean independence E(U|X)=0 is NOT enough (Example 4.1, page 98): it
leaves beta unidentified.  Median independence is what buys
identification while still permitting arbitrary heteroskedasticity.

Condition (b) is a statement about a conditional density, so it is not
decidable from a finite sample.  What is reported here is the
observable evidence for it: how much of the support of X_1 is filled
in within cells of the remaining covariates.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["binidmed", "horowitz_thm4_1_id_median"]


def binidmed(x, beta, ncell=4, nbin=10):
    """Check the Theorem 4.1 conditions for binary-response identification.

    Parameters
    ----------
    x : array-like, (n, d)
    beta : array-like, (d,)
        Scale normalisation is |beta[0]| = 1.
    ncell : int, default 4
        Conditioning cells per remaining covariate, cut at fixed
        quantiles.  Fixed, not chosen from the data.
    nbin : int, default 10
        Bins of the X_1 range used to measure coverage inside a cell.

    Returns
    -------
    RichResult
        payload keys: identified, conda, condb, condscale, rank, dim,
        minsv, coverage, ncells, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    b = np.asarray(beta, dtype=float).ravel()
    if X.shape[1] != b.size and X.shape[0] == b.size:
        X = X.T
    n, d = X.shape

    condscale = bool(abs(abs(float(b[0])) - 1.0) <= 1e-12)

    sv = np.linalg.svd(X, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-12)) if sv.size and sv[0] > 0 else 0
    minsv = float(sv[-1]) if sv.size else 0.0
    conda = bool(rank == d)

    # (b) coverage of the support of X_1 inside cells of Xtilde
    x1 = X[:, 0]
    lo, hi = float(np.min(x1)), float(np.max(x1))
    if d == 1:
        cellid = np.zeros(n, dtype=float)
    else:
        cellid = np.zeros(n, dtype=float)
        mult = 1.0
        for j in range(1, d):
            cuts = np.quantile(X[:, j], np.linspace(0.0, 1.0, ncell + 1))
            idx = np.zeros(n)
            for k in range(1, ncell):
                idx = idx + (X[:, j] >= float(cuts[k])).astype(float)
            cellid = cellid + mult * idx
            mult = mult * ncell
    cells = np.unique(cellid)
    edges = np.linspace(lo, hi, nbin + 1)
    filled = []
    for c in cells:
        sel = cellid == c
        if int(np.sum(sel.astype(float))) == 0:
            continue
        v = x1[sel]
        hit = 0
        for k in range(nbin):
            a, bb = float(edges[k]), float(edges[k + 1])
            inb = (v >= a) & (v <= bb) if k == nbin - 1 else (v >= a) & (v < bb)
            hit += 1 if bool(np.any(inb)) else 0
        filled.append(hit / float(nbin))
    coverage = float(min(filled)) if filled else 0.0
    condb = bool(coverage >= 1.0)

    identified = bool(conda and condb and condscale)
    return RichResult(
        title="Binary-response identification (Theorem 4.1)",
        payload={"identified": identified, "conda": conda, "condb": condb,
                 "condscale": condscale, "rank": rank, "dim": d,
                 "minsv": minsv, "coverage": coverage,
                 "ncells": int(cells.size), "n": n,
                 "method": "Horowitz (2009) Theorem 4.1, median independence"},
    )


horowitz_thm4_1_id_median = binidmed


def cheatsheet():
    return "hrzt41: binary-response identification under median independence"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    m = 40
    g = np.linspace(-3, 3, m)
    X = np.column_stack([np.tile(g, m),
                         np.repeat(np.linspace(-1, 1, m), m)])
    r = binidmed(X, [1.0, 0.5])
    assert r["conda"] and r["condscale"], r
    assert r["coverage"] == 1.0, r["coverage"]
    assert not binidmed(np.column_stack([X[:, 0], 2 * X[:, 0]]),
                        [1.0, 0.5])["conda"]
    print("ok", r["coverage"])
