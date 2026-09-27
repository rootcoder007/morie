"""L-BFGS-B: limited-memory BFGS for bound-constrained minimisation.

Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm for bound
constrained optimization. SIAM Journal on Scientific Computing 16, 1190-1208.
"""

import math

from ._qncore import dot, num_grad, wolfe
from ._richresult import RichResult


def _ssum(it):
    # plain left-to-right summation: sum() of floats is compensated from Python 3.12 on, which
    # would make results depend on the Python version and differ from the R arm
    s = 0.0
    for v in it:
        s += v
    return s


__all__ = ["lbfgsb_minimize"]

_EPS = 2.220446049250313e-16


def _solve(A, b):
    # Gaussian elimination with partial pivoting (small dense systems)
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if M[c][c] == 0:
            raise ZeroDivisionError("singular system")
        for r in range(c + 1, n):
            t = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= t * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - _ssum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def _inverse(A):
    n = len(A)
    cols = [_solve(A, [float(i == j) for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _compact(S, Y, theta):
    """W (n x 2k, rows w_i) and M (2k x 2k) of B = theta I - W M W' (Byrd et al. eqs. 3.2-3.4)."""
    k = len(S)
    if k == 0:
        return None, None
    n = len(S[0])
    W = [[Y[j][i] for j in range(k)] + [theta * S[j][i] for j in range(k)] for i in range(n)]
    SY = [[dot(S[i], Y[j]) for j in range(k)] for i in range(k)]
    SS = [[dot(S[i], S[j]) for j in range(k)] for i in range(k)]
    K = [[0.0] * (2 * k) for _ in range(2 * k)]
    for i in range(k):
        K[i][i] = -SY[i][i]
        for j in range(k):
            if i > j:
                K[k + i][j] = SY[i][j]  # L
                K[j][k + i] = SY[i][j]  # L'
            K[k + i][k + j] = theta * SS[i][j]
    return W, _inverse(K)


def _mv(M, v):
    return [dot(r, v) for r in M]


def _cauchy(x, g, lo, hi, theta, W, M):
    """Generalized Cauchy point, Byrd et al. (1995) Algorithm CP. Returns (xc, c)."""
    n = len(x)
    t = [math.inf] * n
    d = [0.0] * n
    for i in range(n):
        if g[i] < 0:
            t[i] = (x[i] - hi[i]) / g[i]
        elif g[i] > 0:
            t[i] = (x[i] - lo[i]) / g[i]
        d[i] = 0.0 if t[i] == 0 else -g[i]
    k2 = 0 if W is None else len(M)
    p = [_ssum(W[i][j] * d[i] for i in range(n)) for j in range(k2)]
    c = [0.0] * k2
    fp = -dot(d, d)
    fpp = -theta * fp - (dot(p, _mv(M, p)) if k2 else 0.0)
    xc = list(x)
    if fp >= 0:
        return xc, c
    dtm = -fp / fpp if fpp > 0 else math.inf
    free = sorted((t[i], i) for i in range(n) if t[i] > 0 and t[i] < math.inf)
    told = 0.0
    q = 0
    while q < len(free):
        tb, b = free[q]
        dt = tb - told
        if dtm < dt:
            break
        xc[b] = hi[b] if d[b] > 0 else lo[b]
        zb = xc[b] - x[b]
        c = [ci + dt * pi for ci, pi in zip(c, p)]
        gb = g[b]
        if k2:
            wb = W[b]
            Mwb = _mv(M, wb)
            fp = fp + dt * fpp + gb * gb + theta * gb * zb - gb * dot(Mwb, c)
            fpp = fpp - theta * gb * gb - 2 * gb * dot(Mwb, p) - gb * gb * dot(wb, Mwb)
            p = [pi + gb * wi for pi, wi in zip(p, wb)]
        else:
            fp = fp + dt * fpp + gb * gb + theta * gb * zb
            fpp = fpp - theta * gb * gb
        d[b] = 0.0
        told = tb
        q += 1
        if fpp <= 0:
            dtm = 0.0 if fp >= 0 else math.inf
            if fp >= 0:
                break
        else:
            dtm = -fp / fpp
    if q < len(free) or any(t[i] == math.inf and d[i] != 0 for i in range(n)):
        dtm = max(dtm, 0.0)
        if dtm == math.inf:
            dtm = 0.0
        told += dtm
        for i in range(n):
            if d[i] != 0:
                xc[i] = x[i] + told * d[i]
        c = [ci + dtm * pi for ci, pi in zip(c, p)]
    return xc, c


def _subspace(x, g, xc, c, lo, hi, theta, W, M):
    """Direct primal subspace minimisation (Byrd et al. 1995, Sec. 5.1) with backtracking into the box."""
    n = len(x)
    Z = [i for i in range(n) if lo[i] < xc[i] < hi[i]]
    if not Z:
        return list(xc)
    if W is None:
        r = [g[i] + theta * (xc[i] - x[i]) for i in Z]
        du = [-v / theta for v in r]
    else:
        Mc = _mv(M, c)
        r = [g[i] + theta * (xc[i] - x[i]) - dot(W[i], Mc) for i in Z]
        WZ = [W[i] for i in Z]
        k2 = len(M)
        v = _mv(M, [_ssum(WZ[a][j] * r[a] for a in range(len(Z))) for j in range(k2)])
        WtW = [[_ssum(WZ[a][i] * WZ[a][j] for a in range(len(Z))) for j in range(k2)] for i in range(k2)]
        MWtW = [[dot(M[i], [WtW[q][j] for q in range(k2)]) for j in range(k2)] for i in range(k2)]
        N = [[float(i == j) - MWtW[i][j] / theta for j in range(k2)] for i in range(k2)]
        v = _solve(N, v)
        du = [-r[a] / theta - dot(WZ[a], v) / (theta * theta) for a in range(len(Z))]
    alpha = 1.0
    for a, i in enumerate(Z):
        if du[a] > 0:
            alpha = min(alpha, (hi[i] - xc[i]) / du[a])
        elif du[a] < 0:
            alpha = min(alpha, (lo[i] - xc[i]) / du[a])
    xb = list(xc)
    for a, i in enumerate(Z):
        xb[i] = xc[i] + alpha * du[a]
    return xb


def lbfgsb_minimize(f, x0, grad=None, lower=None, upper=None, m=10, pgtol=1e-8, factr=1e7, max_iter=1000):
    r"""Minimise f subject to lower <= x <= upper by L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995).

    Each iteration: the generalized Cauchy point along the projected steepest-descent
    path of the limited-memory quadratic model B = theta I - W M W' (Algorithm CP);
    direct primal minimisation of the model over the variables free at that point,
    backtracked into the box; a strong-Wolfe line search along the resulting
    direction with maximal step 1 (feasible by convexity of the box); the pair
    (s, y) is kept when s'y > eps y'y, the oldest dropped beyond m, and
    theta = y'y / s'y. Stops when the projected gradient max |P(x - g) - x| <= pgtol
    or the relative reduction (f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= factr eps
    (the conventions of R's optim, method "L-BFGS-B").

    Parameters
    ----------
    f : callable
    x0 : sequence of float
        Projected onto the box first.
    grad : callable, optional
        Gradient; central differences when omitted.
    lower, upper : sequence, optional
        Bounds (default unbounded; use +-inf for a free side).
    m : int
        Number of stored correction pairs.
    pgtol, factr : float
    max_iter : int

    Returns
    -------
    RichResult
        Keys: x, fun, grad, projected_gradient, n_iter, n_fev, converged, message.

    References
    ----------
    Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). SIAM Journal on Scientific Computing 16, 1190-1208.

    Examples
    --------
    >>> rb = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    >>> r = lbfgsb_minimize(rb, [-1.2, 1.0], lower=[-2, -2], upper=[0.5, 2])
    >>> [round(v, 6) for v in r["x"]]
    [0.5, 0.25]
    """
    x = [float(v) for v in x0]
    n = len(x)
    lo = [-math.inf] * n if lower is None else [float(v) for v in lower]
    hi = [math.inf] * n if upper is None else [float(v) for v in upper]
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError("lower must not exceed upper")
    x = [min(max(v, a), b) for v, a, b in zip(x, lo, hi)]
    gr = grad if grad is not None else (lambda z: num_grad(f, z))
    fx, g = float(f(x)), [float(v) for v in gr(x)]
    nfev = 1

    def pg_norm(x, g):
        return max((abs(min(max(xi - gi, a), b) - xi) for xi, gi, a, b in zip(x, g, lo, hi)), default=0.0)

    S, Y, theta = [], [], 1.0
    it, msg = 0, "max_iter reached"
    conv = pg_norm(x, g) <= pgtol
    if conv:
        msg = "projected gradient below pgtol"
    while not conv and it < max_iter:
        W, M = _compact(S, Y, theta)
        xc, c = _cauchy(x, g, lo, hi, theta, W, M)
        xb = _subspace(x, g, xc, c, lo, hi, theta, W, M)
        d = [u - v for u, v in zip(xb, x)]
        if dot(d, g) >= 0:  # model direction not downhill: drop the memory and retry once
            if S:
                S, Y, theta = [], [], 1.0
                continue
            msg = "no descent direction"
            break
        cache = {}

        def phi(a, d=d, x=x, cache=cache):
            xa = [min(max(xi + a * di, lb), ub) for xi, di, lb, ub in zip(x, d, lo, hi)]
            fa, ga = float(f(xa)), [float(v) for v in gr(xa)]
            cache[a] = (xa, fa, ga)
            return fa, dot(ga, d)

        a1 = min(1.0, 1.0 / math.sqrt(dot(d, d))) if it == 0 and not S else 1.0
        a, _, _, ne = wolfe(phi, a1=a1, a_max=1.0)
        nfev += ne
        if a == 0.0:
            if S:
                S, Y, theta = [], [], 1.0
                continue
            msg = "line search failed"
            break
        xn, fn, gn = cache[a]
        it += 1
        s = [u - v for u, v in zip(xn, x)]
        y = [u - v for u, v in zip(gn, g)]
        sy, yy = dot(s, y), dot(y, y)
        if sy > _EPS * yy:
            S.append(s)
            Y.append(y)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
            theta = yy / sy
        rel = (fx - fn) / max(abs(fx), abs(fn), 1.0)
        x, fx, g = xn, fn, gn
        if pg_norm(x, g) <= pgtol:
            conv, msg = True, "projected gradient below pgtol"
        elif rel <= factr * _EPS:
            conv, msg = True, "relative reduction of f below factr * epsmch"
    return RichResult(
        title="L-BFGS-B minimisation",
        summary_lines=[("f", fx), ("iterations", it), ("message", msg)],
        payload={
            "x": x,
            "fun": fx,
            "grad": g,
            "projected_gradient": pg_norm(x, g),
            "n_iter": it,
            "n_fev": nfev,
            "converged": conv,
            "message": msg,
        },
    )


def cheatsheet():
    return "lbfgsb: L-BFGS-B bound-constrained limited-memory quasi-Newton (Byrd, Lu, Nocedal and Zhu 1995)"
