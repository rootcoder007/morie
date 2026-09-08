# morie.fn -- function file (rootcoder007/morie)
"""Least median of squares regression.

Rousseeuw, P. J. (1984), "Least Median of Squares Regression",
*Journal of the American Statistical Association* 79(388), 871-880.
The PDF is a scan with no text layer, so every page below was read as
a rendered image.

Equation (1.8), p. 872, defines the estimator:

    minimize_theta  med_i r_i^2.

Theorem 1, p. 872: if p > 1 and the observations are in general
position, the breakdown point of the LMS method is

    ([n/2] - p + 2) / n,

where [r] is the largest integer <= r.  That formula is reported in
the payload and asserted against a hand evaluation.

Theorem 2, p. 873: for p = 1 with all x_i = 1, so that the sample
reduces to (y_i), the LMS location is the midpoint of the shortest
half -- the smallest of y_{h:n} - y_{1:n}, ..., y_{n:n} - y_{n-h+1:n}
with h = [n/2] + 1.  Corollary 1, p. 873: if at least
n - [n/2] + p - 1 of the observations satisfy y_i = x_i theta exactly
and are in general position, the LMS solution equals theta WHATEVER
the other observations are.  Both are used as anchors.

Equation (2.2), p. 874, gives the scale estimate

    S = 1.483 c(n, p) m_T,        m_T^2 = min_theta med_i r_i^2,

where 1/Phi^{-1}(.75) = 1.483.  The paper does NOT give a formula for
the finite-sample correction c(n, p): it says only that work is in
progress to determine it empirically, that it exceeds 1, that it
converges to 1 as n grows, and it quotes the single value
c(20, 6) = 1.8.  So c defaults to 1 here, which is the paper's own
stated large-n limit, and is exposed as a parameter rather than
invented.

ALGORITHM.  The paper's own algorithm for simple regression, p. 874:
for each value of the slope a the intercept subproblem
m_a^2 = min_b med_i ((y_i - a x_i) - b)^2 is solved immediately by the
location algorithm, that is by the shortest half of the partial
residuals.  This implementation does the same in general: it
enumerates the elemental subsets that determine the non-intercept
coefficients exactly, and for each one places the intercept at the
midpoint of the shortest half of the partial residuals.  When the
design has no intercept column the partial-residual step is skipped.
Subsets are enumerated in lexicographic order, not drawn at random, so
both language arms examine the same candidates in the same order.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _rousscore as R
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["least_median_squares"]


def _med_sq(r):
    """med_i r_i^2, the objective of equation (1.8)."""
    sq = sorted(x * x for x in r)
    n = len(sq)
    return sq[n // 2] if n % 2 == 1 else 0.5 * (sq[n // 2 - 1] + sq[n // 2])


def _has_intercept(Xm, n, p):
    for j in range(p):
        allone = True
        for i in range(n):
            if Xm[i][j] != 1.0:
                allone = False
                break
        if allone:
            return j
    return -1


def least_median_squares(y, X, c_np=1.0, max_subsets=200000):
    """LMS regression, equation (1.8) of Rousseeuw (1984).

    Parameters
    ----------
    y : array-like
        n responses.
    X : array-like
        n-by-p design matrix; include the column of ones yourself if
        the model has an intercept.
    c_np : float
        The finite-sample correction c(n, p) of equation (2.2).
        Defaults to 1, the paper's stated large-n limit; the paper
        gives no formula for it.
    max_subsets : int
        Refuse rather than enumerate more than this many elemental subsets.

    Returns
    -------
    estimate : the minimised med r^2
    coef     : the p coefficients
    scale    : S = 1.483 c(n, p) sqrt(estimate), equation (2.2)
    residual : the residuals at the solution
    breakdown : ([n/2] - p + 2) / n, Theorem 1
    intercept_col : which design column was recognised as the intercept, or -1
    """
    yy = k.vec(y)
    Xm = k.mat(X)
    n = len(yy)
    if n == 0:
        raise ValueError("least_median_squares: y is empty")
    if k.nrow(Xm) != n:
        raise ValueError("least_median_squares: X must have one row per response")
    p = k.ncol(Xm)
    if p == 0:
        raise ValueError("least_median_squares: X has no columns")
    if n < p:
        raise ValueError("least_median_squares: need at least p observations")
    total = R.nchoosek(n, p)
    if total > max_subsets:
        raise ValueError("least_median_squares: %d elemental subsets exceeds max_subsets" % total)
    ic = _has_intercept(Xm, n, p)
    hloc = n // 2 + 1
    best = None
    for J in R.combos(n, p):
        A = [[Xm[i][j] for j in range(p)] for i in J]
        b = [yy[i] for i in J]
        th = R.lusolve(A, b)
        if th is None:
            continue
        if ic >= 0:
            # Solve the intercept subproblem exactly, as on p. 874: the
            # best b is the midpoint of the shortest half of the partial
            # residuals.  This is Theorem 2 applied to y - X theta with
            # the intercept coefficient removed.
            part = []
            for i in range(n):
                s = yy[i]
                for j in range(p):
                    if j != ic:
                        s -= th[j] * Xm[i][j]
                part.append(s)
            a0, w, srt = R.shortest_half(part, hloc)
            th = list(th)
            th[ic] = 0.5 * (srt[a0] + srt[a0 + hloc - 1])
        res = []
        for i in range(n):
            s = yy[i]
            for j in range(p):
                s -= th[j] * Xm[i][j]
            res.append(s)
        obj = _med_sq(res)
        if best is None or obj < best[0]:
            best = (obj, list(th), res)
    if best is None:
        raise ValueError("least_median_squares: every elemental subset was singular")
    obj, th, res = best
    scale = 1.483 * float(c_np) * math.sqrt(obj)
    breakdown = float(n // 2 - p + 2) / float(n)
    return RichResult(
        title="Least median of squares regression",
        summary_lines=[("n", n), ("p", p), ("med r^2", obj), ("scale", scale)],
        payload={
            "estimate": obj,
            "coef": th,
            "scale": scale,
            "residual": res,
            "breakdown": breakdown,
            "intercept_col": float(ic),
            "c_np": float(c_np),
            "n": n,
            "p": p,
            "n_subsets": total,
            "method": "Rousseeuw (1984) eq. (1.8) min med r^2, elemental fits with the p. 874 shortest-half intercept step; scale eq. (2.2)",
        },
    )


def cheatsheet():
    return "lmsreg: least median of squares regression"
