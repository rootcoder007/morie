"""Transformation model MLE."""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as stats


def _solve(A, b):
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("trmle: the information matrix is singular")
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                if f != 0.0:
                    M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def trmle(
    y: np.ndarray,
    X: np.ndarray,
    *,
    n_basis: int = 5,
) -> dict:
    r"""
    Transformation model by maximum likelihood (most likely transformation).

    Fits :math:`h(Y) = X'\beta + \varepsilon`, :math:`\varepsilon \sim
    N(0, 1)`, i.e. :math:`P(Y \le y \mid x) = \Phi(h(y) - x'\beta)`, with
    the unknown increasing transformation :math:`h(y) = a(y)'\theta`
    written in a Bernstein polynomial basis of order ``n_basis - 1`` on
    the observed range of ``y``.  The error variance is fixed at one and
    ``X`` carries NO intercept: both are absorbed by ``h``, which is what
    makes the model identified -- a free scale on :math:`h`, :math:`\beta`
    and :math:`\sigma` together, or a free intercept beside :math:`h`'s
    own, is not.

    The log-likelihood
    :math:`\sum_i \log\phi(h(y_i) - x_i'\beta) + \log h'(y_i)`
    is concave in :math:`(\theta, \beta)` (a log-normal density of a
    linear form plus the log of a linear form).  :math:`h` is kept
    increasing by requiring :math:`\theta_0 \le \dots \le \theta_M`, and
    the bound-constrained concave problem is solved by projected Newton.
    Standard errors are the inverse observed information in
    :math:`(\theta, \beta)`, as ``mlt`` reports them.

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Covariates (n, p), without an intercept column.
    n_basis : int
        Number of Bernstein basis functions (polynomial order + 1).

    Returns
    -------
    dict
        ``beta``, ``se``, ``t_stat``, ``pval``, ``basis_coefs`` (theta),
        ``log_likelihood``, ``n_obs``.

    References
    ----------
    Hothorn, T., Moest, L. & Buehlmann, P. (2018). Most likely
    transformations. Scandinavian Journal of Statistics 45(1), 110-134,
    doi:10.1111/sjos.12291; the R package ``mlt`` (``ctm`` with
    ``negative = TRUE``) fits the same model.
    Horowitz, J. L. (1996). Semiparametric estimation of a regression
    model with an unknown transformation of the dependent variable.
    Econometrica 64, 103-137 (the fully nonparametric alternative).
    """
    yv = [float(v) for v in np.asarray(y, dtype=float).ravel().tolist()]
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    Xm = [[float(v) for v in row] for row in Xa.tolist()]
    n, p = len(Xm), len(Xm[0])
    if len(yv) != n:
        raise ValueError("y and X must have same n.")
    K = int(n_basis)
    if K < 2:
        raise ValueError("n_basis must be at least 2.")
    if n < p + K + 2:
        raise ValueError("Insufficient observations.")
    for j in range(p):
        col = [r[j] for r in Xm]
        if max(col) - min(col) == 0.0:
            raise ValueError("X must not contain a constant column; the "
                             "intercept is part of h.")
    lo_, hi_ = min(yv), max(yv)
    if hi_ <= lo_:
        raise ValueError("y is constant.")
    M = K - 1
    w = hi_ - lo_
    s = [(v - lo_) / w for v in yv]
    B = [[math.comb(M, k) * si ** k * (1 - si) ** (M - k) for k in range(K)]
         for si in s]
    # h'(y) = sum_k theta_k D_k(y), D_k = (M/w)(b^{M-1}_{k-1} - b^{M-1}_k)
    D = []
    for si in s:
        lower = [math.comb(M - 1, k) * si ** k * (1 - si) ** (M - 1 - k)
                 for k in range(M)]
        D.append([(M / w) * ((lower[k - 1] if k >= 1 else 0.0)
                             - (lower[k] if k < M else 0.0))
                  for k in range(K)])
    q = K + p

    def parts(par):
        th, be = par[:K], par[K:]
        ll, g = 0.0, [0.0] * q
        H = [[0.0] * q for _ in range(q)]
        for i in range(n):
            h = sum(a * b for a, b in zip(B[i], th))
            hp = sum(a * b for a, b in zip(D[i], th))
            if hp <= 0.0:
                return None
            z = h - sum(a * b for a, b in zip(Xm[i], be))
            ll += -0.5 * z * z - 0.5 * math.log(2 * math.pi) + math.log(hp)
            # d z / d par: (B_i, -x_i); d hp / d par: (D_i, 0)
            dz = B[i] + [-v for v in Xm[i]]
            dh = D[i] + [0.0] * p
            for a in range(q):
                g[a] += -z * dz[a] + dh[a] / hp
                for c in range(q):
                    H[a][c] -= dz[a] * dz[c] + dh[a] * dh[c] / (hp * hp)
        return ll, g, H

    # h must be increasing, so theta is: theta_0 free, theta_k =
    # theta_{k-1} + delta_k with delta_k >= 0 (mlt's ui = "increasing").
    # The problem stays concave in (theta_0, delta, beta) with simple
    # bounds, solved by projected Newton: variables at their bound with
    # the gradient pushing outward are held, Newton on the rest, and the
    # trial point is projected back onto delta >= 0.
    def to_theta(u):
        th, acc = [], u[0]
        th.append(acc)
        for k in range(1, K):
            acc += u[k]
            th.append(acc)
        return th + u[K:]

    def uparts(u):
        r = parts(to_theta(u))
        if r is None:
            return None
        ll_, g_, H_ = r
        # theta_j = u_0 + sum_{i=1..j} u_i: d theta_j / d u_i = 1 for i <= j
        J = [[1.0 if (c == 0 or c <= r_) else 0.0 for c in range(K)]
             for r_ in range(K)]
        gu = [sum(J[r_][c] * g_[r_] for r_ in range(K)) for c in range(K)] + g_[K:]
        Hu = [[0.0] * q for _ in range(q)]
        for a_ in range(q):
            for c in range(q):
                if a_ < K and c < K:
                    Hu[a_][c] = sum(J[r1][a_] * J[r2][c] * H_[r1][r2]
                                    for r1 in range(K) for r2 in range(K))
                elif a_ < K:
                    Hu[a_][c] = sum(J[r1][a_] * H_[r1][c] for r1 in range(K))
                elif c < K:
                    Hu[a_][c] = sum(J[r2][c] * H_[a_][r2] for r2 in range(K))
                else:
                    Hu[a_][c] = H_[a_][c]
        return ll_, gu, Hu

    mu = sum(yv) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in yv) / n) or 1.0
    th0 = [((lo_ + k / M * w) - mu) / sd for k in range(K)]
    u = [th0[0]] + [th0[k] - th0[k - 1] for k in range(1, K)] + [0.0] * p
    cur = uparts(u)
    for _ in range(500):
        ll, g, H = cur
        act = [j for j in range(1, K) if u[j] <= 1e-14 and g[j] <= 0.0]
        free = [j for j in range(q) if j not in act]
        sub = [[-H[a_][c] for c in free] for a_ in free]
        sf = _solve(sub, [g[a_] for a_ in free])
        step = [0.0] * q
        for a_, v in zip(free, sf):
            step[a_] = v
        t = 1.0
        nxt = None
        while t >= 1e-14:
            cand = [u[j] + t * step[j] for j in range(q)]
            for j in range(1, K):
                cand[j] = max(cand[j], 0.0)
            r_ = uparts(cand)
            if r_ is not None and r_[0] >= ll - 1e-13:
                nxt = r_
                break
            t *= 0.5
        if nxt is None:
            break
        moved = max(abs(c - v) for c, v in zip(cand, u))
        u, cur = cand, nxt
        if moved < 1e-13:
            break
    par = to_theta(u)
    cur = parts(par)
    ll, g, H = cur
    q_inv = [_solve([[-v for v in row] for row in H],
                    [1.0 if r == c else 0.0 for r in range(q)])
             for c in range(q)]
    beta = par[K:]
    se = [math.sqrt(max(q_inv[K + j][K + j], 0.0)) for j in range(p)]
    t_stat = [b / s_ if s_ > 0 else float("nan") for b, s_ in zip(beta, se)]
    pval = [float(2 * stats.norm.sf(abs(v))) for v in t_stat]
    return {
        "beta": beta,
        "se": se,
        "t_stat": t_stat,
        "pval": pval,
        "basis_coefs": par[:K],
        "log_likelihood": float(ll),
        "n_obs": n,
    }


trmle_fn = trmle


def cheatsheet() -> str:
    return "trmle({y, X}) -> most likely transformation model, Phi(h(y) - x'b)."
