"""Nonlinear least squares by Gauss-Newton with step halving."""

import math

from ._richresult import RichResult
from .linsys import _householder_ls

__all__ = ["nonlinear_least_squares"]


def _jacobian(model, x, theta):
    # central differences, step cbrt(eps) * max(|theta_j|, 1)
    cols = []
    for j in range(len(theta)):
        h = 6.055454452393343e-06 * max(abs(theta[j]), 1.0)
        tp, tm = list(theta), list(theta)
        tp[j] += h
        tm[j] -= h
        cols.append([(model(xi, tp) - model(xi, tm)) / (2 * h) for xi in x])
    return [[cols[j][i] for j in range(len(theta))] for i in range(len(x))]


def _inverse(a):
    n = len(a)
    m = [list(r) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(a)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(m[i][k]))
        m[k], m[p] = m[p], m[k]
        piv = m[k][k]
        m[k] = [v / piv for v in m[k]]
        for i in range(n):
            if i != k:
                f = m[i][k]
                m[i] = [u - f * v for u, v in zip(m[i], m[k])]
    return [r[n:] for r in m]


def nonlinear_least_squares(model, x, y, start, tol=1e-8, max_iter=200):
    r"""Minimise :math:`\sum_i (y_i - f(x_i, \theta))^2` by Gauss-Newton, as ``nls``.

    Each step solves the linearised problem :math:`\min_\delta \|J\delta - r\|`
    by Householder QR and halves the step until the residual sum of squares
    falls. Convergence uses the relative-offset criterion of ``nls`` (Bates &
    Watts 1988): :math:`\sqrt{\|Q_1'r\|^2/p}\,/\,\sqrt{\|Q_2'r\|^2/(n-p)} <` ``tol``.
    Standard errors are :math:`\sqrt{\mathrm{diag}\,\hat\sigma^2(J'J)^{-1}}`
    with :math:`\hat\sigma^2 = RSS/(n-p)` (Hedderich, Sachs & Reynarowych 2023,
    Sec. 3.7.12, which fits :math:`a + bx + cx^2` and :math:`ab^x` with ``nls``).

    Parameters
    ----------
    model : callable
        ``model(x_i, theta) -> float``.
    x : sequence
        Covariate values (any objects ``model`` accepts).
    y : sequence of float
    start : sequence of float
        Starting values.
    tol : float
    max_iter : int

    Returns
    -------
    RichResult
        ``coefficients``, ``se``, ``rss``, ``sigma``, ``fitted``, ``iterations``,
        ``converged``.

    References
    ----------
    Bates, D. M. & Watts, D. G. (1988). Nonlinear Regression Analysis and
    Its Applications. Wiley, ch. 2.
    """
    yy = [float(v) for v in y]
    theta = [float(v) for v in start]
    n, p = len(yy), len(theta)
    if n != len(x) or n <= p:
        raise ValueError("need len(x) == len(y) > number of parameters")

    def resid(t):
        return [yi - model(xi, t) for xi, yi in zip(x, yy)]

    r = resid(theta)
    rss = math.fsum(v * v for v in r)
    converged, it = False, 0
    for _ in range(max_iter):
        it += 1
        J = _jacobian(model, x, theta)
        delta, rperp = _householder_ls(J, r)
        if rperp <= 0 or math.sqrt(max(rss - rperp, 0.0) / p) / math.sqrt(rperp / (n - p)) < tol:
            converged = True
            break
        fac = 1.0
        while fac >= 1 / 1024:
            cand = [t + fac * d for t, d in zip(theta, delta)]
            rc = resid(cand)
            rssc = math.fsum(v * v for v in rc)
            if rssc < rss:
                theta, r, rss = cand, rc, rssc
                break
            fac /= 2
        else:
            # the predicted Gauss-Newton decrease is below the rounding error
            # of rss itself, so no step can lower it: theta is the minimiser
            if rss - rperp <= 4 * n * 2.220446049250313e-16 * rss:
                converged = True
                break
            raise ValueError("step factor reduced below 1/1024 without reducing the residual sum of squares")
    J = _jacobian(model, x, theta)
    jtj = [[sum(J[i][a] * J[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    s2 = rss / (n - p)
    inv = _inverse(jtj)
    se = [math.sqrt(s2 * inv[j][j]) for j in range(p)]
    return RichResult(
        title="Nonlinear least squares (Gauss-Newton)",
        summary_lines=[("coefficients", theta), ("rss", rss)],
        payload={
            "coefficients": theta,
            "se": se,
            "rss": rss,
            "sigma": math.sqrt(s2),
            "fitted": [model(xi, theta) for xi in x],
            "iterations": it,
            "converged": converged,
        },
    )


def cheatsheet():
    return "nlsgn: Gauss-Newton with step halving; nls relative-offset convergence; se from s^2 (J'J)^-1"
