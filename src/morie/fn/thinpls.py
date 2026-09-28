"""Thin-plate spline interpolation and smoothing in two dimensions (Duchon 1977; Wahba 1990)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve
from ._richresult import RichResult

__all__ = ["thin_plate_spline"]


def _phi(r):
    return r * r * math.log(r) / (8.0 * math.pi) if r > 0.0 else 0.0


def thin_plate_spline(x, y, *, lam: float = 0.0, newdata=None) -> RichResult:
    r"""Thin-plate spline fit and prediction.

    Minimises ``sum_i (y_i - f(x_i))^2 + lam * J(f)`` with the bending
    energy ``J(f) = int int (f_uu^2 + 2 f_uv^2 + f_vv^2)``; the solution is
    ``f(s) = a_0 + a_1 u + a_2 v + sum_i w_i phi(|s - x_i|)`` with
    ``phi(r) = r^2 log(r) / (8 pi)``, found from

    .. math::

        \begin{pmatrix} K + \lambda I & P \\ P^T & 0 \end{pmatrix}
        \begin{pmatrix} w \\ a \end{pmatrix} =
        \begin{pmatrix} y \\ 0 \end{pmatrix},

    ``K_ij = phi(|x_i - x_j|)``, ``P = [1, x]`` (Duchon 1977; Wahba 1990,
    ch. 2).  ``lam = 0`` interpolates.  This is ``fields::Tps`` with the
    same ``lambda`` and ``scale.type = "unscaled"``.

    :param x: (n, 2) knot locations.
    :param y: Values (n,).
    :param lam: Smoothing parameter.
    :param newdata: Optional (m, 2) prediction locations.
    :return: :class:`RichResult` with ``w``, ``a``, ``fitted``,
        ``residuals``, ``df`` (trace of the hat matrix) and ``predicted``.

    References
    ----------
    Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in
    Sobolev spaces. In *Constructive Theory of Functions of Several
    Variables*, 85-100. Springer, Berlin.
    Wahba, G. (1990). *Spline Models for Observational Data*. SIAM,
    Philadelphia.

    Examples
    --------
    >>> pts = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
    >>> r = thin_plate_spline(pts, [0.0, 1.0, 1.0, 2.0, 1.5], newdata=[(0.5, 0.5), (0.25, 0.75)])
    >>> [round(v, 6) for v in r.predicted]
    [1.5, 1.294285]
    """
    X = [(float(a), float(b)) for a, b in np.asarray(x, dtype=float).tolist()]
    yv = [float(v) for v in np.asarray(y, dtype=float).tolist()]
    n = len(X)
    if len(yv) != n or n < 3:
        raise ValueError("need at least three knots with one value each")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    m = n + 3
    A = [[0.0] * m for _ in range(m)]
    for i in range(n):
        for j in range(n):
            A[i][j] = _phi(math.hypot(X[i][0] - X[j][0], X[i][1] - X[j][1])) + (lam if i == j else 0.0)
        row = (1.0, X[i][0], X[i][1])
        for k in range(3):
            A[i][n + k] = row[k]
            A[n + k][i] = row[k]
    sol = [float(v) for v in solve(A, yv + [0.0, 0.0, 0.0])]
    w, a = sol[:n], sol[n:]

    def f(s):
        return (
            a[0]
            + a[1] * s[0]
            + a[2] * s[1]
            + sum(w[i] * _phi(math.hypot(s[0] - X[i][0], s[1] - X[i][1])) for i in range(n))
        )

    fitted = [f(p) for p in X]
    # hat matrix: fitted = y - lam * w, so its trace is n - lam * tr(A^-1 restricted to the knots)
    if lam > 0:
        Ainv = inverse(A)
        tr = n - lam * sum(float(Ainv[k][k]) for k in range(n))
    else:
        tr = float(n)
    pred = None
    if newdata is not None:
        pred = [f((float(a_), float(b_))) for a_, b_ in np.asarray(newdata, dtype=float).tolist()]
    return RichResult(
        payload={
            "w": w,
            "a": a,
            "fitted": fitted,
            "residuals": [yv[i] - fitted[i] for i in range(n)],
            "df": tr,
            "predicted": pred,
            "lam": float(lam),
        }
    )


thinpls = thin_plate_spline


def cheatsheet() -> str:
    return "thinpls: 2-D thin-plate spline interpolation / smoothing (fields::Tps)."
