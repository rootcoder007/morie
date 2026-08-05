# morie.fn -- function file (rootcoder007/morie)
"""Sample-split TMLE for a data-adaptively selected covariate subset."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_subset_selection"]


def tmle_subset_selection(y, D, X):
    """Targeted ATE under a covariate subset chosen on a held-out split.

    Selecting the adjustment set and then estimating on the same rows
    makes the target parameter itself a function of the data, and the
    usual influence-curve SE is then wrong -- it prices the estimator
    but not the selection.  The fix used here is the data-adaptive
    target-parameter framework: the sample is split deterministically
    into even and odd indices, the subset is chosen on the ODD half, and
    the parameter is defined and estimated on the EVEN half.  Given the
    split, the target is a fixed (if random) parameter, and the
    influence curve computed on the estimation half is valid for it
    without a selection correction.

    Selection ranks covariates by the absolute correlation between the
    covariate and the treatment-residualised outcome on the selection
    half, and keeps the top ``ceil(p/2)``.  The estimation half then
    runs an ordinary point-treatment TMLE on the kept columns.

    The reported ``se`` is therefore the standard error of a HALF-sample
    estimator: it does not shrink by using the selection rows, and that
    is the price of an honest interval.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Candidate covariates.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n_selected``, ``n_est``, ``n``.

    References
    ----------
    Hubbard, A. E., Kherad-Pajouh, S. & van der Laan, M. J. (2016).
    Statistical inference for data adaptive target parameters.
    International Journal of Biostatistics 12(1):3-19.
    doi:10.1515/ijb-2015-0013.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    n = len(yv)
    if n < 8 or len(Dv) != n:
        raise ValueError("tmle_subset_selection: y and D must share one length >= 8")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_subset_selection: X must have one row per subject")
    p = len(Xm[0])
    sel = [i for i in range(n) if i % 2 == 1]
    est = [i for i in range(n) if i % 2 == 0]
    b, _, res, _ = S.ols([[1.0, Dv[i]] for i in sel], [yv[i] for i in sel])
    score = []
    for j in range(p):
        col = [Xm[i][j] for i in sel]
        mc = sum(col) / len(col)
        mr = sum(res) / len(res)
        num = sum((col[k] - mc) * (res[k] - mr) for k in range(len(sel)))
        dc = math.sqrt(sum((v - mc) ** 2 for v in col))
        dr = math.sqrt(sum((v - mr) ** 2 for v in res))
        score.append(abs(num) / (dc * dr) if dc > 0 and dr > 0 else 0.0)
    k = (p + 1) // 2
    order = sorted(range(p), key=lambda j: (-score[j], j))
    keep = sorted(order[:k])
    ne = len(est)
    W = [[1.0] + [Xm[i][j] for j in keep] for i in est]
    ye = [yv[i] for i in est]
    de = [Dv[i] for i in est]
    out = S.tmle(ye, de, W)
    return RichResult(payload={
        "estimate": out["psi"], "se": out["se"], "eps": out["eps"],
        "n_selected": float(len(keep)), "n_est": float(ne), "n": n,
        "method": "Sample-split TMLE on a data-adaptively selected covariate subset"})


def cheatsheet():
    return "tmlsbs: split-sample TMLE with data-adaptive covariate selection."
