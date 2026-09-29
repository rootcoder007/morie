"""Spatial frailty survival model"""

import math

from ._containers import SpatialResult
from ._qpcore import inverse, solve


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _prep(time, event, X, region, W):
    t = _vec(time)
    dl = [int(v) for v in event]
    n = len(t)
    if X is None:
        Xm = [[] for _ in range(n)]
    else:
        Xm = [
            [float(v) for v in (r if hasattr(r, "__len__") else [r])]
            for r in (X.tolist() if hasattr(X, "tolist") else X)
        ]
    reg = [int(v) for v in region]
    A = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    R = len(A)
    Q = [[(sum(A[i]) if i == j else 0.0) - A[i][j] for j in range(R)] for i in range(R)]
    return t, dl, Xm, reg, Q


def _newton(grad, obj, x0, tol=1e-10, maxit=200):
    """Newton-Raphson on an analytic gradient with a central-difference Hessian and step halving."""
    x = list(x0)
    k = len(x)
    f = obj(x)
    it = 0
    for _ in range(maxit):
        it += 1
        g = grad(x)
        H = []
        for j in range(k):
            h = 1e-5 * max(1.0, abs(x[j]))
            xp, xm = list(x), list(x)
            xp[j] += h
            xm[j] -= h
            gp, gm = grad(xp), grad(xm)
            H.append([(gp[i] - gm[i]) / (2 * h) for i in range(k)])
        H = [[0.5 * (H[i][j] + H[j][i]) for j in range(k)] for i in range(k)]
        step = solve([[-v for v in r] for r in H], g)
        s = 1.0
        while True:
            xn = [a + s * b for a, b in zip(x, step)]
            fn = obj(xn)
            if fn == fn and fn >= f - 1e-12 * abs(f):
                break
            s *= 0.5
            if s < 1e-12:
                xn, fn = x, f
                break
        done = max(abs(s * b) for b in step) < tol
        x, f = xn, fn
        if done:
            break
    return x, f, H, it


def spatial_frailty(time, event, X, region, W, *, tau=1.0, tol=1e-10):
    r"""Weibull proportional-hazards model with intrinsic-CAR regional frailties by penalised likelihood.

    ``h_i(t) = rho t^(rho-1) exp(x_i' beta + u_r(i))`` with the log-frailties
    ``u`` of the ``R`` regions tied by the ICAR penalty ``(tau/2) u'(D - W)u``
    (``W`` the regional adjacency; its null space, the constant, plays the
    intercept). The penalised log-likelihood

    ``sum_i [d_i (log rho + (rho - 1) log t_i + eta_i) - t_i^rho e^eta_i] -
    (tau/2) u'Qu``

    is maximised over ``(log rho, beta, u)`` by Newton-Raphson on its
    analytic gradient (the h-likelihood / penalised-likelihood fit of a
    spatial frailty model at fixed precision ``tau``; Banerjee, Wall and
    Carlin 2003 give the Bayesian version; Therneau and Grambsch 2000, ch.
    9, the penalised-likelihood view of frailties). Standard errors are from
    the inverse negative Hessian. ``statistic`` is the penalised
    log-likelihood.

    References
    ----------
    Banerjee, S., Wall, M. M. and Carlin, B. P. (2003). Frailty modeling for
    spatially correlated survival data, with application to infant mortality
    in Minnesota. *Biostatistics* 4, 123-142.
    Therneau, T. M. and Grambsch, P. M. (2000). *Modeling Survival Data:
    Extending the Cox Model*. Springer.

    Examples
    --------
    >>> import math
    >>> W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> reg = [i % 3 for i in range(30)]
    >>> x = [[math.sin(i)] for i in range(30)]
    >>> t = [0.5 + (i * 7 % 11) / 5 * math.exp(-0.5 * x[i][0] - 0.3 * reg[i]) for i in range(30)]
    >>> ev = [0 if i % 5 == 0 else 1 for i in range(30)]
    >>> r = spatial_frailty(t, ev, x, reg, W)
    >>> round(r.extra["shape"], 8)
    3.06543534
    """
    t, dl, Xm, reg, Q = _prep(time, event, X, region, W)
    n, p, R = len(t), len(Xm[0]), len(Q)
    lt = [math.log(v) for v in t]

    def parts(th):
        a, b, u = th[0], th[1 : 1 + p], th[1 + p :]
        rho = math.exp(a)
        eta = [sum(Xm[i][k] * b[k] for k in range(p)) + u[reg[i]] for i in range(n)]
        Hc = [math.exp(rho * lt[i] + eta[i]) for i in range(n)]
        return a, rho, u, eta, Hc

    def obj(th):
        a, rho, u, eta, Hc = parts(th)
        ll = sum(dl[i] * (a + (rho - 1.0) * lt[i] + eta[i]) - Hc[i] for i in range(n))
        return ll - 0.5 * tau * sum(u[i] * Q[i][j] * u[j] for i in range(R) for j in range(R))

    def grad(th):
        a, rho, u, eta, Hc = parts(th)
        ga = sum(dl[i] * (1.0 + rho * lt[i]) - Hc[i] * rho * lt[i] for i in range(n))
        gb = [sum((dl[i] - Hc[i]) * Xm[i][k] for i in range(n)) for k in range(p)]
        gu = [
            sum(dl[i] - Hc[i] for i in range(n) if reg[i] == r) - tau * sum(Q[r][j] * u[j] for j in range(R))
            for r in range(R)
        ]
        return [ga] + gb + gu

    x, f, H, it = _newton(grad, obj, [0.0] * (1 + p + R), tol=tol)
    V = inverse([[-v for v in r] for r in H])
    se = [math.sqrt(V[i][i]) if V[i][i] > 0 else float("nan") for i in range(len(x))]
    return SpatialResult(
        name="zesfr",
        statistic=f,
        extra={
            "shape": math.exp(x[0]),
            "coefficients": x[1 : 1 + p],
            "frailties": x[1 + p :],
            "se": se,
            "iterations": it,
            "tau": tau,
        },
    )


spat = spatial_frailty


def cheatsheet() -> str:
    return "spatial_frailty(time, event, X, region, W, tau) -> Weibull PH with ICAR frailties, penalised likelihood."


# compact alias per ledger/NAMING.md
spatialfrailty = spatial_frailty
