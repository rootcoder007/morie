# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Ordinary least squares in ILR coordinates (compositional regression).

The composition-valued response is carried into real coordinates by the
isometric log-ratio map, ordinary least squares is run there, and the
fit is carried back to the simplex by the inverse map:

    Y_ilr = ilr(Y),     Y_ilr = X B + E,     Yhat = ilr^-1(X B).

This is legitimate precisely because ilr is an isometry of the Aitchison
geometry onto Euclidean space, so the least-squares criterion in
coordinates is the Aitchison-distance criterion on the simplex; that
equivalence is the point of Pawlowsky-Glahn, Egozcue and Tolosana-Delgado
(2015), *Modeling and Analysis of Compositional Data*, Wiley, and it is
checked here as an anchor rather than assumed.

The default basis is the sequential binary partition of Egozcue,
Pawlowsky-Glahn, Mateu-Figueras and Barcelo-Vidal (2003), "Isometric
logratio transformations for compositional data analysis", Mathematical
Geology 35(3), 279-300, doi:10.1023/a:1023818214614 (verified against
Crossref), equation (11) as rendered from the page image.

X is used exactly as supplied; if an intercept is wanted, put a column
of ones in it.  Nothing is added silently.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["compositional_regression"]


def _basis(D):
    V = [[0.0] * (D - 1) for _ in range(D)]
    for i in range(1, D):
        c = math.sqrt(i / (i + 1.0))
        for j in range(i):
            V[j][i - 1] = c / i
        V[i][i - 1] = -c
    return V


def _ilr_row(x, V):
    lg = [math.log(v) for v in x]
    m = sum(lg) / len(lg)
    z = [v - m for v in lg]
    out = []
    for i in range(len(V[0])):
        s = 0.0
        for j in range(len(V)):
            s += V[j][i] * z[j]
        out.append(s)
    return out


def _inv_ilr(y, V):
    lx = []
    for j in range(len(V)):
        s = 0.0
        for i in range(len(y)):
            s += V[j][i] * y[i]
        lx.append(s)
    m = max(lx)
    e = [math.exp(v - m) for v in lx]
    t = sum(e)
    return [v / t for v in e]


def compositional_regression(X, Y_comp, V=None):
    """Least squares of ilr(Y) on X.

    Parameters
    ----------
    X : array-like
        N-by-p design matrix, used verbatim.
    Y_comp : array-like
        N-by-D matrix of strictly positive compositions.
    V : optional
        D-by-(D-1) contrast matrix; defaults to the Egozcue et al. (2003)
        sequential binary partition.

    Returns
    -------
    beta : p-by-(D-1) coefficients in ilr coordinates
    fitted : N-by-(D-1) fitted coordinates
    resid : N-by-(D-1) residual coordinates
    fitted_comp : the fits carried back to the simplex
    sse : the residual sum of squares, equal to the total squared
        Aitchison distance between Y and its fit
    """
    Xm = [[float(v) for v in r] for r in X]
    Ym = [[float(v) for v in r] for r in Y_comp]
    N = len(Xm)
    if N == 0 or len(Ym) == 0:
        raise ValueError("compositional_regression: no observations")
    if len(Ym) != N:
        raise ValueError("compositional_regression: X and Y_comp have different row counts")
    p = len(Xm[0])
    D = len(Ym[0])
    if D < 2:
        raise ValueError("compositional_regression: a composition needs at least 2 parts")
    for r in Xm:
        if len(r) != p:
            raise ValueError("compositional_regression: X is ragged")
    for r in Ym:
        if len(r) != D:
            raise ValueError("compositional_regression: Y_comp is ragged")
        for v in r:
            if not (v > 0.0):
                raise ValueError("compositional_regression: every part of Y_comp must be positive")
    if N < p:
        raise ValueError("compositional_regression: fewer observations than columns of X")
    Vm = _basis(D) if V is None else [[float(a) for a in r] for r in V]
    Yi = [_ilr_row(r, Vm) for r in Ym]
    q = len(Vm[0])
    beta = [[0.0] * q for _ in range(p)]
    for c in range(q):
        col = [Yi[i][c] for i in range(N)]
        b = k.lstsq(Xm, col, ridge=0.0)
        for j in range(p):
            beta[j][c] = float(b[j])
    fitted = []
    resid = []
    sse = 0.0
    for i in range(N):
        f = []
        for c in range(q):
            s = 0.0
            for j in range(p):
                s += Xm[i][j] * beta[j][c]
            f.append(s)
        fitted.append(f)
        e = [Yi[i][c] - f[c] for c in range(q)]
        resid.append(e)
        for v in e:
            sse += v * v
    fitted_comp = [_inv_ilr(f, Vm) for f in fitted]
    return RichResult(
        title="Compositional regression in ilr coordinates",
        summary_lines=[("N", N), ("p", p), ("D", D)],
        payload={
            "beta": beta,
            "fitted": fitted,
            "resid": resid,
            "fitted_comp": fitted_comp,
            "Y_ilr": Yi,
            "sse": sse,
            "estimate": beta[0][0],
            "N": N,
            "p": p,
            "D": D,
            "method": "OLS of ilr(Y) on X in the Egozcue et al. (2003) SBP basis",
        },
    )


def cheatsheet():
    return "aitcrg: OLS in ILR coordinates (compositional regression)"


# compact alias per ledger/NAMING.md
compositionalregression = compositional_regression
