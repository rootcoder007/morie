# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric Moran eigenvector spatial filtering (Tiefelsdorf and Griffith 2007)."""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as stats
from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = ["moran_eigenvector_filter"]


def _mm(A, B):
    Bt = list(zip(*B))
    return [[ssum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def _annihilator(X):
    n, p = len(X), len(X[0])
    Xc = list(zip(*X))
    XtXi = inverse([[ssum(a * b for a, b in zip(Xc[i], Xc[j])) for j in range(p)] for i in range(p)])
    XtXi = [[float(v) for v in r] for r in XtXi]
    H = _mm(_mm(X, XtXi), [list(c) for c in Xc])
    return [[(1.0 if i == j else 0.0) - H[i][j] for j in range(n)] for i in range(n)]


def _moran_moments(M, S, df):
    MSM = _mm(_mm(M, S), M)
    n = len(M)
    t1 = ssum(MSM[i][i] for i in range(n))
    t2 = ssum(MSM[i][j] * MSM[j][i] for i in range(n) for j in range(n))
    return t1 / df, 2.0 * (df * t2 - t1 * t1) / (df * df * (df + 2.0)), MSM


def _quad(a, S, b):
    n = len(a)
    return ssum(a[i] * ssum(S[i][j] * b[j] for j in range(n)) for i in range(n))


def moran_eigenvector_filter(
    y,
    X,
    adjacency,
    *,
    tol: float = 0.1,
    zerovalue: float = 1e-4,
    alpha: float | None = None,
) -> RichResult:
    r"""Stepwise Moran eigenvector spatial filter for a linear regression.

    Following Tiefelsdorf and Griffith (2007) as ``spatialreg::SpatialFiltering``
    (defaults: style ``C``, symmetric, ``ExactEV = FALSE``): the binary
    neighbour matrix is symmetrised and scaled to sum to ``n``, projected
    with ``M_X = I - X (X'X)^{-1} X'`` to ``M_X S M_X`` and eigendecomposed.
    Among the eigenvectors whose eigenvalue has the sign of the residual
    autocorrelation (``|value| > zerovalue``), each step adds the one that
    brings the residual Moran's I, ``e'Se / e'e``, closest to its
    expectation, in units of the standard deviation under the current
    design (``E = tr(MSM)/df``, ``Var = 2 (df tr(MSM^2) - tr(MSM)^2) /
    (df^2 (df + 2))``, ``df = n - k``).  Selection stops once
    ``|z| < tol`` or ``|z|`` grows (with ``alpha``: once the two-sided
    p-value of ``z`` reaches ``alpha``).

    :param y: Response (n,).
    :param X: Design matrix (n, p), including any intercept column.
    :param adjacency: (n, n) neighbour indicator; symmetrised internally.
    :param tol: Stop when ``|z|`` of the residual Moran's I is below it.
    :param zerovalue: Eigenvalues within it of zero are never candidates.
    :param alpha: Optional p-value stopping rule in place of ``tol``.
    :return: :class:`RichResult` with ``selection`` (rows ``step``,
        ``evec`` (1-based index in decreasing eigenvalue order), ``eval``,
        ``moran``, ``z``, ``p_value``, ``r2``, ``gamma``), ``vectors`` (the
        selected eigenvectors as columns), ``coefficients`` (``X`` then the
        eigenvectors), ``fitted`` and ``stop_reason`` (``tol``, ``alpha``,
        ``inversion`` when ``|z|`` grew, or ``exhausted``).  Eigenvector signs are arbitrary, so
        a ``gamma`` can flip sign with its vector; fitted values do not.

    References
    ----------
    Tiefelsdorf, M. and Griffith, D. A. (2007). Semiparametric filtering
    of spatial autocorrelation: the eigenvector approach. *Environment and
    Planning A*, 39(5), 1193-1221.

    Examples
    --------
    >>> A = [[1 if abs(i - j) == 1 else 0 for j in range(12)] for i in range(12)]
    >>> y = [1.0, 1.4, 2.2, 2.9, 3.1, 2.6, 2.0, 1.1, 0.8, 1.5, 2.4, 2.7]
    >>> X = [[1.0, v] for v in (0.2, 0.5, 0.1, 0.9, 0.4, 0.3, 0.8, 0.6, 0.7, 0.05, 0.35, 0.55)]
    >>> r = moran_eigenvector_filter(y, X, A)
    >>> [row["evec"] for row in r.selection], r.stop_reason
    ([0, 1, 3], 'inversion')
    """
    y = [float(v) for v in np.asarray(y, dtype=float).tolist()]
    X = [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    A = [[float(v != 0) for v in r] for r in np.asarray(adjacency, dtype=float).tolist()]
    n = len(y)
    if len(X) != n or len(A) != n or any(len(r) != n for r in A):
        raise ValueError("y, X and adjacency must share n")
    for i in range(n):
        A[i][i] = 0.0
    rs = [sum(r) for r in A]
    s0 = sum(rs)
    eff = sum(1 for v in rs if v > 0)
    C = [[v * eff / s0 for v in r] for r in A]
    S = [[0.5 * (C[i][j] + C[j][i]) for j in range(n)] for i in range(n)]
    tot = ssum(v for r in S for v in r)
    S = [[n / tot * v for v in r] for r in S]
    MX = _annihilator(X)
    P = _mm(_mm(MX, S), MX)
    P = [[0.5 * (P[i][j] + P[j][i]) for j in range(n)] for i in range(n)]
    w, V = np.linalg.eigh(np.asarray(P))
    w = [float(v) for v in w]
    V = [[float(v) for v in r] for r in V.tolist()]
    order = sorted(range(n), key=lambda k: -w[k])
    val = [w[k] for k in order]
    vec = [[V[i][k] for i in range(n)] for k in order]
    p = len(X[0])
    ybar = sum(y) / n
    tss = ssum((v - ybar) ** 2 for v in y)

    def pval(z):
        return float(2.0 * stats.norm.sf(abs(z)))

    M = MX
    E, Var, MSM = _moran_moments(M, S, n - p)
    cyMy = _quad(y, M, y)
    i0 = _quad(y, MSM, y) / cyMy
    z0 = (i0 - E) / math.sqrt(Var)
    rows = [{"step": 0, "evec": 0, "eval": 0.0, "moran": i0, "z": z0, "p_value": pval(z0), "r2": 1.0 - cyMy / tss}]
    sel = [(1 if v > abs(zerovalue) else 0) - (1 if v < -abs(zerovalue) else 0) for v in val]
    Xc = [list(c) for c in zip(*X)]
    XtXi = [
        [float(v) for v in r]
        for r in inverse([[ssum(a * b for a, b in zip(Xc[i], Xc[j])) for j in range(p)] for i in range(p)])
    ]
    coef = [ssum(XtXi[a][b] * ssum(Xc[b][i] * y[i] for i in range(n)) for b in range(p)) for a in range(p)]
    base = [y[i] - ssum(X[i][k] * coef[k] for k in range(p)) for i in range(n)]
    acsign = -1 if _quad(base, S, base) / ssum(v * v for v in base) < 0 else 1
    gam = [ssum(vec[j][i] * y[i] for i in range(n)) / ssum(v * v for v in vec[j]) for j in range(n)]
    chosen = []
    cur = list(base)
    old = math.inf
    reason = "exhausted"
    for _step in range(1, n + 1):
        best, bz, bmi = -1, math.inf, 0.0
        for j in range(n):
            if sel[j] != acsign:
                continue
            res = [cur[i] - vec[j][i] * gam[j] for i in range(n)]
            mi = _quad(res, S, res) / ssum(v * v for v in res)
            if abs((mi - E) / math.sqrt(Var)) < bz:
                best, bz, bmi = j, abs((mi - E) / math.sqrt(Var)), mi
        if best < 0:
            break
        chosen.append(best)
        cur = [cur[i] - vec[best][i] * gam[best] for i in range(n)]
        Xa = [X[i] + [vec[k][i] for k in chosen] for i in range(n)]
        M = _annihilator(Xa)
        E, Var, MSM = _moran_moments(M, S, n - len(Xa[0]))
        z = (bmi - E) / math.sqrt(Var)
        rows.append(
            {
                "step": len(chosen),
                "evec": best + 1,
                "eval": val[best],
                "moran": bmi,
                "z": z,
                "p_value": pval(z),
                "r2": 1.0 - _quad(y, M, y) / tss,
            }
        )
        sel[best] = 0
        if (abs(z) < tol) if alpha is None else (pval(z) >= alpha):
            reason = "tol" if alpha is None else "alpha"
            break
        if abs(z) > abs(old):  # inversion: z moved away from zero
            reason = "inversion"
            break
        old = z
    Xa = [X[i] + [vec[k][i] for k in chosen] for i in range(n)]
    q = len(Xa[0])
    Ac = [list(c) for c in zip(*Xa)]
    AtAi = [
        [float(v) for v in r]
        for r in inverse([[ssum(a * b for a, b in zip(Ac[i], Ac[j])) for j in range(q)] for i in range(q)])
    ]
    Aty = [ssum(Ac[k][i] * y[i] for i in range(n)) for k in range(q)]
    bg = [ssum(AtAi[a][b] * Aty[b] for b in range(q)) for a in range(q)]
    for t, row in enumerate(rows):
        row["gamma"] = 0.0 if t == 0 else bg[p + t - 1]
    return RichResult(
        payload={
            "selection": rows,
            "vectors": [[vec[k][i] for k in chosen] for i in range(n)],
            "coefficients": bg,
            "fitted": [ssum(Xa[i][k] * bg[k] for k in range(q)) for i in range(n)],
            "stop_reason": reason,
        }
    )


def cheatsheet() -> str:
    return (
        "moran_eigenvector_filter(y, X, A) -> stepwise Moran eigenvector spatial filter (spatialreg::SpatialFiltering)."
    )
