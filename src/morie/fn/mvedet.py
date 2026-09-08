# morie.fn -- function file (rootcoder007/morie)
"""Minimum volume ellipsoid.

Rousseeuw, P. J. (1985), "Multivariate estimation with high breakdown
point", in *Mathematical Statistics and Applications*, Vol. B,
Reidel, 283-297.  The MVE is the ellipsoid of smallest volume that
covers at least h of the n observations; its centre estimates location
and its shape, suitably scaled, estimates scatter.  Like the MCD it
attains the maximal breakdown point at h = [(n + p + 1) / 2].

Computation follows the standard resampling scheme: for each
(p+1)-subset J take the mean m_J and covariance C_J of J, and inflate
the ellipsoid until it covers h points, which means scaling by the
h-th smallest squared Mahalanobis distance m2_J.  The volume of
{x : (x - m)' C^{-1} (x - m) <= m2} is proportional to
m2^{p/2} sqrt(det C) = sqrt(det(m2 * C)), so minimising the volume is
minimising det(m2_J * C_J) over J.  That determinant is the objective
reported here.

DETERMINISM.  The subsets are enumerated deterministically rather than
drawn at random, so both language arms examine the same candidates in
the same order.  They are taken by an even STRIDE through the
lexicographic enumeration, not from its prefix: the prefix is drawn
almost entirely from the lowest indices, which biases the search
towards whatever happens to sit at the start of the data.  See the
note in _rousscore.combos_stride, where a confusion matrix caught
exactly that failure in the sibling FastMCD module.

The univariate case is a closed form and is this module's anchor: for
p = 1 an ellipsoid is an interval, so the MVE is the SHORTEST interval
containing h points and its centre is that interval's midpoint.  That
is exactly the shortest-half construction Rousseeuw (1984) Theorem 2
gives on p. 873 of *Least Median of Squares Regression*, and it is
computed by sorting, not by this search.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _rousscore as R
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["mve"]


def mve(X, h=None, n_starts=100000):
    """The minimum volume ellipsoid estimator.

    Parameters
    ----------
    X : array-like
        n-by-p data matrix.
    h : int, optional
        Coverage; defaults to [(n + p + 1) / 2].
    n_starts : int
        Cap on the number of (p+1)-subsets enumerated.

    Returns
    -------
    estimate : det(m2 * C) at the optimum, proportional to the squared volume
    center   : the MVE location
    cov      : m2 * C, the scaled scatter matrix
    m2       : the inflation factor, the h-th smallest squared distance
    subset   : the (p+1)-subset that generated the winner
    covered  : the indices covered by the optimal ellipsoid
    """
    Xm = k.mat(X)
    n = k.nrow(Xm)
    if n == 0:
        raise ValueError("mve: X is empty")
    p = k.ncol(Xm)
    if p == 0:
        raise ValueError("mve: X has no columns")
    hh = R.mcd_h(n, p) if h is None else int(h)
    if hh <= p:
        raise ValueError("mve: h must exceed p")
    if hh > n:
        raise ValueError("mve: h cannot exceed the number of observations")
    if n < p + 1:
        raise ValueError("mve: need at least p + 1 observations")
    best = None
    for J in R.combos_stride(n, p + 1, int(n_starts)):
        mu, C = R.meancov(Xm, J)
        dd = R.mahal2(Xm, mu, C)
        if dd is None:
            continue
        order = R.osort(dd)
        m2 = dd[order[hh - 1]]
        dC = R.covdet(C)
        obj = (m2 ** p) * dC
        if best is None or obj < best[0]:
            best = (obj, m2, mu, C, J, sorted(order[:hh]))
    if best is None:
        raise ValueError("mve: every subset was degenerate")
    obj, m2, mu, C, J, covered = best
    Sc = [[C[a][b] * m2 for b in range(p)] for a in range(p)]
    return RichResult(
        title="Minimum volume ellipsoid",
        summary_lines=[("n", n), ("p", p), ("h", hh), ("objective", obj)],
        payload={
            "estimate": obj,
            "center": mu,
            "cov": Sc,
            "cov_raw": C,
            "m2": m2,
            "subset": [float(v) for v in J],
            "covered": [float(v) for v in covered],
            "h": hh,
            "n": n,
            "p": p,
            "method": "Rousseeuw (1985) MVE, lexicographic (p+1)-subsets inflated to cover h points, objective det(m2 * C)",
        },
    )


def cheatsheet():
    return "mvedet: minimum volume ellipsoid"
