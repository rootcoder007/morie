# morie.fn -- function file (rootcoder007/morie)
"""Minimum covariance determinant, by exhaustive enumeration.

Rousseeuw, P. J. (1985), "Multivariate estimation with high breakdown
point", in *Mathematical Statistics and Applications*, Vol. B,
Reidel, 283-297, where the MCD was introduced.  The definition used
here is the one restated by Hubert, Debruyne and Rousseeuw (2018),
"Minimum Covariance Determinant and Extensions", arXiv:1709.07045,
section "Definition", which was read directly:

    the raw MCD with tuning constant n/2 <= h <= n is (mu0, S0) where
    mu0 is the mean of the h observations for which the determinant of
    the sample covariance matrix is as small as possible, and S0 is
    that covariance matrix multiplied by a consistency factor c0.

The same source gives the consistency factor,

    c0 = alpha / F_{chi2_{p+2}}(q_alpha),

with alpha = h/n and q_alpha the alpha-quantile of the chi2_p
distribution, and gives the most robust subset size h = [(n+p+1)/2]
along with the requirement h > p (otherwise every h-subset has a
singular covariance matrix), so n > 2p.

THIS function computes the estimator by its definition: it enumerates
every h-subset.  That is exponential, and the same source says so
plainly -- "the exact MCD estimator is very hard to compute, as it
requires the evaluation of all (n choose h) subsets".  The point of
having it is that it is the ground truth the FastMCD approximation in
module fastm is checked against.  It refuses rather than silently
approximating when the enumeration would be larger than max_subsets.

The univariate case has a closed form the enumeration must reproduce:
for p = 1 the h-subset of smallest variance is CONTIGUOUS in the
sorted sample, so it can be read off directly.  That is this module's
anchor.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _rousscore as R
from . import _s03core as k
from .pchsq import pchisq
from .qchsq import qchisq

from ._richresult import RichResult

__all__ = ["mcd"]


def consistency_factor(h, n, p):
    """c0 = alpha / F_{chi2_{p+2}}(q_alpha), alpha = h/n, q_alpha = chi2_p quantile."""
    alpha = float(h) / float(n)
    if alpha >= 1.0:
        return 1.0
    q = float(qchisq(alpha, p))
    f = float(pchisq(q, p + 2))
    return alpha / f if f > 0.0 else 1.0


def mcd(X, h=None, n_starts=None, max_subsets=200000):
    """The exact minimum covariance determinant estimator.

    Parameters
    ----------
    X : array-like
        n-by-p data matrix.
    h : int, optional
        Subset size.  Defaults to [(n + p + 1) / 2], the most robust
        choice.
    n_starts : ignored
        Accepted so the signature matches the approximate algorithm in
        module fastm; exhaustive enumeration has no starts.
    max_subsets : int
        Refuse rather than enumerate more than this many subsets.

    Returns
    -------
    estimate : the minimised determinant of the raw covariance
    center   : the MCD location
    cov_raw  : the covariance of the best subset, unscaled
    cov      : cov_raw times the consistency factor
    factor   : the consistency factor c0
    subset   : the 0-based indices of the best subset, ascending
    h, n, p, n_subsets
    """
    Xm = k.mat(X)
    n = k.nrow(Xm)
    if n == 0:
        raise ValueError("mcd: X is empty")
    p = k.ncol(Xm)
    if p == 0:
        raise ValueError("mcd: X has no columns")
    hh = R.mcd_h(n, p) if h is None else int(h)
    if hh <= p:
        raise ValueError("mcd: h must exceed p, otherwise every subset is singular")
    if hh > n:
        raise ValueError("mcd: h cannot exceed the number of observations")
    total = R.nchoosek(n, hh)
    if total > max_subsets:
        raise ValueError("mcd: %d subsets exceeds max_subsets; use fastm for the approximate algorithm" % total)
    best_idx = None
    best_det = None
    for idx in R.combos(n, hh):
        mu, S = R.meancov(Xm, idx)
        d = R.covdet(S)
        if best_det is None or d < best_det:
            best_det = d
            best_idx = idx
    mu, S = R.meancov(Xm, best_idx)
    c0 = consistency_factor(hh, n, p)
    Sc = [[S[a][b] * c0 for b in range(p)] for a in range(p)]
    return RichResult(
        title="Minimum covariance determinant (exhaustive)",
        summary_lines=[("n", n), ("p", p), ("h", hh), ("subsets", total), ("det", best_det)],
        payload={
            "estimate": best_det,
            "center": mu,
            "cov_raw": S,
            "cov": Sc,
            "factor": c0,
            "subset": [float(v) for v in best_idx],
            "h": hh,
            "n": n,
            "p": p,
            "n_subsets": total,
            "method": "Rousseeuw (1985) MCD by exhaustive enumeration of all h-subsets; c0 = alpha / F_chi2_{p+2}(q_alpha)",
        },
    )


def cheatsheet():
    return "mcdv: exact minimum covariance determinant by enumeration"
