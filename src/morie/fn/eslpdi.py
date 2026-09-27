# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Product density ICA, ProDenICA (ESL Sec 14.7.4, eqs 14.89-14.96, Algorithm 14.3)."""

import math

from . import _array_core as np
from ._richresult import RichResult
from .eslgam import _spline_smooth

__all__ = ["esl_prodenica"]

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def _inv_sqrt(M):
    """Symmetric inverse square root of a symmetric positive-definite matrix (list of lists)."""
    d, E = np.linalg.eigh(np.asarray(M))
    d, E = list(d), E.tolist()
    if min(d) <= 0:
        raise ValueError("matrix is not positive definite (rank-deficient data?)")
    p = len(d)
    return [[sum(E[i][k] * E[j][k] / math.sqrt(d[k]) for k in range(p)) for j in range(p)] for i in range(p)]


def _matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def _spline_eval(u, f, g, t):
    """Natural cubic spline through (u, f) with second derivatives g: value, first and second derivative at t."""
    m = len(u)
    if t <= u[0] or t >= u[-1]:
        k = 0 if t <= u[0] else m - 1
        h = u[1] - u[0] if k == 0 else u[-1] - u[-2]
        d = (
            (f[1] - f[0]) / h - h * (2 * g[0] + g[1]) / 6
            if k == 0
            else (f[-1] - f[-2]) / h + h * (g[-2] + 2 * g[-1]) / 6
        )
        return f[k] + d * (t - u[k]), d, 0.0
    k = min(int((t - u[0]) / (u[1] - u[0])), m - 2)  # equally spaced grid
    h = u[k + 1] - u[k]
    a, b = t - u[k], u[k + 1] - t
    val = (a * f[k + 1] + b * f[k]) / h - a * b / 6 * ((1 + a / h) * g[k + 1] + (1 + b / h) * g[k])
    d1 = (f[k + 1] - f[k]) / h + ((3 * a * a - h * h) * g[k + 1] - (3 * b * b - h * h) * g[k]) / (6 * h)
    return val, d1, (a * g[k + 1] + b * g[k]) / h


def _fit_tilt(s, L, penalty, widen=1.2, max_iter=100):
    """Penalised Poisson fit of the log tilt g on a grid (14.93-14.94), by IRLS with a cubic smoothing spline."""
    n = len(s)
    lo, hi = min(s), max(s)
    c, half = (lo + hi) / 2, (hi - lo) / 2 * widen
    grid = [c - half + 2 * half * q / (L - 1) for q in range(L)]
    delta = grid[1] - grid[0]
    ys = [0.0] * L
    for v in s:
        ys[min(L - 1, max(0, int((v - grid[0] + delta / 2) / delta)))] += 1.0 / n
    base = [delta * math.exp(-t * t / 2 - _LOG_SQRT_2PI) for t in grid]  # Delta * phi(s*)
    # start as glm does for Poisson counts, mu = y + 0.1
    g = [math.log((y * n + 0.1) / n / b) for y, b in zip(ys, base)]
    for _ in range(max_iter):
        mu = [b * math.exp(v) for b, v in zip(base, g)]
        z = [v + (y - m) / m for v, y, m in zip(g, ys, mu)]
        new, u, f, gam = _spline_smooth(grid, z, 2 * penalty, mu, full=True)
        done = max(abs(a - b) for a, b in zip(new, g)) < 1e-10
        g = new
        if done:
            break
    return u, f, gam


def esl_prodenica(X, penalty=1e-4, L=500, A0=None, max_iter=50, tol=1e-9):
    r"""Product density ICA (ProDenICA; Hastie & Tibshirani 2003, ESL Algorithm 14.3).

    The centred data are whitened symmetrically, :math:`Z = (X - \bar x)\Sigma^{-1/2}`
    (divisor N), and the sources :math:`s_j = a_j^T z` get tilted-Gaussian
    densities :math:`f_j = \phi e^{g_j}` (14.89). Alternate until the
    orthogonal A settles:

    (a) for each j maximise (14.92) over :math:`g_j`, approximated on a grid
        of L points spanning ``1.2`` times the range of :math:`s_j` with
        bin proportions :math:`y^*_\ell` (14.93): the penalised Poisson
        log-likelihood (14.94)
        :math:`\sum_\ell y^*_\ell[\log\phi(s^*_\ell)+g(s^*_\ell)] - \Delta\phi(s^*_\ell)e^{g(s^*_\ell)} - \lambda\int g''^2`,
        a cubic smoothing spline (the book notes cubic is adequate) fitted by
        Newton/IRLS; the unpenalised constant makes :math:`\sum_\ell\Delta\phi e^{\hat g} = 1`;
    (b) one fixed-point step (14.96)
        :math:`a_j \leftarrow E[z\,\hat g_j'(a_j^T z)] - E[\hat g_j''(a_j^T z)]a_j`,
        then :math:`A \leftarrow A(A^TA)^{-1/2}` (:math:`UV^T`).

    Parameters
    ----------
    X : N x p nested sequence
    penalty : float
        :math:`\lambda` in (14.94) (larger = closer to Gaussian).
    L : int
        Grid size.
    A0 : p x p nested sequence, optional
        Starting directions (columns), orthogonalised; default the identity.
    max_iter, tol
        Stop when :math:`1 - \min_j |a_j^{old\,T} a_j^{new}| <` ``tol``.

    Returns
    -------
    RichResult
        ``A`` (orthogonal, columns :math:`a_j`), ``sources`` (:math:`ZA`),
        ``unmixing`` (:math:`\Sigma^{-1/2}A`, so sources = (X - mean) unmixing),
        ``whitening``, ``mean``, ``negentropy`` (C(A), 14.95),
        ``negentropy_path``, ``iterations``, ``converged``.

    References
    ----------
    Hastie, T. & Tibshirani, R. (2003). Independent components analysis
    through product density estimation. NIPS 15, 649-656.

    Examples
    --------
    >>> xs, S = 12345, []
    >>> for _ in range(400):
    ...     u = []
    ...     for _ in range(2):
    ...         xs = xs * 16807 % 2147483647
    ...         u.append(xs / 2147483647)
    ...     S.append([(u[0] - 0.5) * 12 ** 0.5, -math.log(u[1]) - 1])
    >>> X = [[a + 0.6 * b, 0.4 * a + b] for a, b in S]
    >>> r = esl_prodenica(X)
    >>> r["converged"]
    True
    """
    X = [[float(v) for v in x] for x in X]
    n = len(X)
    p = len(X[0]) if n else 0
    if n < 10 or p < 1 or L < 10 or penalty <= 0:
        raise ValueError("need N >= 10, L >= 10 and penalty > 0")
    mean = [sum(c) / n for c in zip(*X)]
    Xc = [[v - m for v, m in zip(x, mean)] for x in X]
    K = _inv_sqrt([[sum(x[i] * x[j] for x in Xc) / n for j in range(p)] for i in range(p)])
    Z = _matmul(Xc, K)
    A = [[float(i == j) for j in range(p)] for i in range(p)] if A0 is None else [[float(v) for v in r] for r in A0]
    A = _matmul(A, _inv_sqrt(_matmul(list(map(list, zip(*A))), A)))
    path, conv, it = [], False, 0
    for _ in range(max_iter):
        it += 1
        S = _matmul(Z, A)
        new_cols, C = [], 0.0
        for j in range(p):
            s = [row[j] for row in S]
            u, f, gam = _fit_tilt(s, L, penalty)
            ev = [_spline_eval(u, f, gam, v) for v in s]
            C += sum(e[0] for e in ev) / n
            m2 = sum(e[2] for e in ev) / n
            new_cols.append([sum(z[c] * e[1] for z, e in zip(Z, ev)) / n - m2 * A[c][j] for c in range(p)])
        path.append(C)
        An = list(map(list, zip(*new_cols)))
        An = _matmul(An, _inv_sqrt(_matmul(list(map(list, zip(*An))), An)))
        moved = 1 - min(abs(sum(A[c][j] * An[c][j] for c in range(p))) for j in range(p))
        A = An
        if moved < tol:
            conv = True
            break
    S = _matmul(Z, A)
    return RichResult(
        title="Product density ICA",
        summary_lines=[("p", p), ("negentropy", path[-1]), ("iterations", it)],
        payload={
            "A": A,
            "sources": S,
            "unmixing": _matmul(K, A),
            "whitening": K,
            "mean": mean,
            "negentropy": path[-1],
            "negentropy_path": path,
            "iterations": it,
            "converged": conv,
        },
    )
