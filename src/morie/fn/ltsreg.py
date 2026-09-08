# morie.fn -- function file (rootcoder007/morie)
"""Least trimmed squares regression.

Rousseeuw, P. J. (1984), "Least Median of Squares Regression",
*Journal of the American Statistical Association* 79(388), 871-880,
Section 4 "Related approaches", p. 876, equation (4.1), read from a
rendered page image because the PDF is a scan with no text layer:

    minimize_theta  sum_{i=1}^{h} (r^2)_{i:n},

"where (r^2)_{1:n} <= ... <= (r^2)_{n:n} are the ordered squared
residuals.  If h = [n/2] + 1 is chosen, the breakdown point of
Theorem 1 is obtained, and for h = [n/2] + [(p+1)/2], the result of
Remark 1 holds.  In general, h may depend on some trimming proportion
alpha, for instance by means of h = [n(1 - alpha)] + 1."

The default here is the maximal-breakdown choice of Remark 1,
h = [n/2] + [(p+1)/2], whose breakdown point (p. 873) is
([(n - p)/2] + 1)/n.  Both breakdown formulas are reported.

The same page states the asymptotics: LTS converges like n^{-1/2},
unlike the LMS which converges like n^{-1/3}, which is the reason
Section 4 introduces it.

ALGORITHM.  Concentration, the direct analogue of the MCD C-step:
from a trial fit, refit by ordinary least squares on the h
observations with the smallest squared residuals.  The objective
cannot increase -- the new fit minimises the sum of squares over that
same subset, and the next subset is by construction the h smallest
under the new fit -- so the iteration terminates.  That monotonicity
is asserted as an anchor rather than assumed.

DETERMINISM.  Starts are elemental p-subsets enumerated in
lexicographic order, not drawn at random, so both language arms visit
the same candidates in the same sequence.

At h = n nothing is trimmed and the estimator is exactly ordinary
least squares.  That is the module's closed-form anchor, checked
against the normal equations solved by a different route.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _rousscore as R
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["least_trimmed_squares"]


def _fit(Xm, yy, idx, p):
    """Ordinary least squares on a subset, by the normal equations."""
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for i in idx:
        for a in range(p):
            for c in range(p):
                A[a][c] += Xm[i][a] * Xm[i][c]
            b[a] += Xm[i][a] * yy[i]
    return R.lusolve(A, b)


def _obj(Xm, yy, th, n, p, h):
    """The h smallest squared residuals, their sum, and their indices."""
    sq = []
    for i in range(n):
        s = yy[i]
        for j in range(p):
            s -= th[j] * Xm[i][j]
        sq.append(s * s)
    order = R.osort(sq)
    idx = sorted(order[:h])
    tot = 0.0
    for i in idx:
        tot += sq[i]
    return tot, idx, sq


def least_trimmed_squares(y, X, h=None, max_starts=200000, max_iter=100):
    """LTS regression, equation (4.1) of Rousseeuw (1984).

    Parameters
    ----------
    y : array-like
        n responses.
    X : array-like
        n-by-p design matrix; include the intercept column yourself.
    h : int, optional
        Number of retained residuals.  Defaults to the maximal-breakdown
        choice of Remark 1, [n/2] + [(p+1)/2].
    max_starts : int
        Cap on the elemental subsets enumerated.
    max_iter : int
        Cap on the concentration steps per start.

    Returns
    -------
    estimate : the minimised sum of the h smallest squared residuals
    coef     : the p coefficients
    subset   : the retained indices
    residual : residuals at the solution
    objectives : the objective chain of the winning start, non-increasing
    breakdown_remark1 : ([(n - p)/2] + 1) / n
    breakdown_theorem1 : ([n/2] - p + 2) / n, the value at h = [n/2] + 1
    """
    yy = k.vec(y)
    Xm = k.mat(X)
    n = len(yy)
    if n == 0:
        raise ValueError("least_trimmed_squares: y is empty")
    if k.nrow(Xm) != n:
        raise ValueError("least_trimmed_squares: X must have one row per response")
    p = k.ncol(Xm)
    if p == 0:
        raise ValueError("least_trimmed_squares: X has no columns")
    hh = R.trimmed_h(n, p) if h is None else int(h)
    if hh < p:
        raise ValueError("least_trimmed_squares: h must be at least p")
    if hh > n:
        raise ValueError("least_trimmed_squares: h cannot exceed the number of observations")
    total = R.nchoosek(n, p)
    if total > max_starts:
        raise ValueError("least_trimmed_squares: %d elemental subsets exceeds max_starts" % total)
    best = None
    for J in R.combos(n, p):
        A = [[Xm[i][j] for j in range(p)] for i in J]
        b = [yy[i] for i in J]
        th = R.lusolve(A, b)
        if th is None:
            continue
        chain = []
        idx = None
        for _ in range(int(max_iter)):
            tot, idx, sq = _obj(Xm, yy, th, n, p, hh)
            chain.append(tot)
            nth = _fit(Xm, yy, idx, p)
            if nth is None:
                break
            ntot, nidx, nsq = _obj(Xm, yy, nth, n, p, hh)
            if nidx == idx:
                th = nth
                chain.append(ntot)
                idx = nidx
                break
            th = nth
            idx = nidx
        tot, idx, sq = _obj(Xm, yy, th, n, p, hh)
        if best is None or tot < best[0]:
            best = (tot, list(th), idx, chain)
    if best is None:
        raise ValueError("least_trimmed_squares: every elemental subset was singular")
    tot, th, idx, chain = best
    res = []
    for i in range(n):
        s = yy[i]
        for j in range(p):
            s -= th[j] * Xm[i][j]
        res.append(s)
    return RichResult(
        title="Least trimmed squares regression",
        summary_lines=[("n", n), ("p", p), ("h", hh), ("objective", tot)],
        payload={
            "estimate": tot,
            "coef": th,
            "subset": [float(v) for v in idx],
            "residual": res,
            "objectives": chain,
            "breakdown_remark1": float((n - p) // 2 + 1) / float(n),
            "breakdown_theorem1": float(n // 2 - p + 2) / float(n),
            "h": hh,
            "n": n,
            "p": p,
            "method": "Rousseeuw (1984) eq. (4.1) min sum of h smallest squared residuals, concentration from lexicographic elemental starts",
        },
    )


def cheatsheet():
    return "ltsreg: least trimmed squares regression"
