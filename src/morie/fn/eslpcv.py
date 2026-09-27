"""Principal curves by the Hastie-Stuetzle algorithm (ESL sec 14.5.2)."""

import math

from ._richresult import RichResult
from .eslgam import _spline_smooth

__all__ = ["esl_principal_curve"]


def _project(x, pts):
    """Closest point on the polyline ``pts`` to x: returns (arc length, squared distance, point)."""
    best, acc = None, 0.0
    for a, b in zip(pts, pts[1:]):
        d = [bb - aa for aa, bb in zip(a, b)]
        L2 = sum(v * v for v in d)
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, sum((xx - aa) * dd for xx, aa, dd in zip(x, a, d)) / L2))
        q = [aa + t * dd for aa, dd in zip(a, d)]
        dist = sum((xx - qq) ** 2 for xx, qq in zip(x, q))
        if best is None or dist < best[1] - 1e-15:
            best = (acc + t * math.sqrt(L2), dist, q)
        acc += math.sqrt(L2)
    return best


def esl_principal_curve(X, penalty=1.0, max_iter=10, tol=1e-3):
    r"""Fit a principal curve :math:`f(\lambda) = E(X|\lambda_f(X) = \lambda)` (ESL eqs 14.61-14.62).

    Hastie & Stuetzle (1989): start from the first principal-component line;
    then alternate (a) the conditional-expectation step, smoothing each
    coordinate :math:`x_{ij}` against the current projection indices
    :math:`\lambda_i` with a cubic smoothing spline of the given ``penalty``, and
    (b) the projection step :math:`\lambda_i = \arg\min_\lambda\|x_i - \hat f(\lambda)\|^2`
    onto the polyline through the fitted points, re-parametrised by arc
    length; stop when the total squared distance changes by less than
    ``tol`` relatively (or the points lie on the curve). The iteration is
    not guaranteed to converge and can settle into a small 2-cycle, so the
    defaults follow princurve (``thresh = 0.001``, ``maxit = 10``).

    Parameters
    ----------
    X : N x p nested sequence
    penalty : float
        Smoothing-spline penalty in the arc-length scale (smaller = wigglier).
    max_iter, tol
        Iteration controls.

    Returns
    -------
    RichResult
        ``lambda`` (arc-length index per point), ``fitted`` (curve point per
        observation), ``curve`` (fitted points ordered by lambda),
        ``distance`` (sum of squared distances), ``distance_path``,
        ``iterations``, ``converged``.

    References
    ----------
    Hastie, T. & Stuetzle, W. (1989). Principal curves. JASA 84, 502-516.
    """
    from . import _array_core as np

    rows = [[float(v) for v in r] for r in X]
    N, p = len(rows), len(rows[0])
    if N < 4 or penalty < 0:
        raise ValueError("need at least 4 points and penalty >= 0")
    mu = [sum(r[j] for r in rows) / N for j in range(p)]
    C = np.asarray([[r[j] - mu[j] for j in range(p)] for r in rows])
    v = [float(t) for t in np.linalg.svd(C, full_matrices=False)[2][0]]
    lam = [sum((r[j] - mu[j]) * v[j] for j in range(p)) for r in rows]
    lo = min(lam)
    lam = [t - lo for t in lam]
    fitted = [[mu[j] + (t + lo) * v[j] for j in range(p)] for t in lam]
    dist = sum(sum((a - b) ** 2 for a, b in zip(r, f)) for r, f in zip(rows, fitted))
    path, it, conv = [dist], 0, False
    for _ in range(max_iter):
        it += 1
        cols = [_spline_smooth(lam, [r[j] for r in rows], penalty) for j in range(p)]
        fitted = [[cols[j][i] for j in range(p)] for i in range(N)]
        order = sorted(range(N), key=lambda i: (lam[i], i))
        pts = []
        for i in order:
            if not pts or pts[-1] != fitted[i]:
                pts.append(fitted[i])
        proj = [_project(r, pts) for r in rows]
        lam = [pr[0] for pr in proj]
        fitted = [pr[2] for pr in proj]
        new = sum(pr[1] for pr in proj)
        path.append(new)
        if abs(dist - new) <= tol * max(dist, 1e-300) or new < 1e-20:
            dist = new
            conv = True
            break
        dist = new
    order = sorted(range(N), key=lambda i: (lam[i], i))
    return RichResult(
        title="Principal curve",
        summary_lines=[("distance", dist), ("iterations", it)],
        payload={
            "lambda": lam,
            "fitted": fitted,
            "curve": [fitted[i] for i in order],
            "distance": dist,
            "distance_path": path,
            "iterations": it,
            "converged": conv,
        },
    )


def cheatsheet():
    return "eslpcv: Hastie-Stuetzle: smooth each coordinate on lambda, project onto the polyline (arc length), repeat"
