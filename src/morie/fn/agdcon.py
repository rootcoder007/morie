# morie.fn -- function file (rootcoder007/morie)
"""Accelerated projected gradient on a box."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["agdproj", "agd_constrained"]


def agdproj(X, y, lower=None, upper=None, steps=100, lipschitz=None):
    """FISTA with the prox of a box indicator, i.e. Euclidean projection.

    Minimises f(b) = 0.5 ||X b - y||^2 over the box lower <= b <= upper.
    The accelerated proximal-gradient iteration is unchanged from the
    unconstrained case; only the proximal map differs.  For the indicator
    of a closed convex set the proximal map is the Euclidean projection
    onto that set, which for a box is a component-wise clamp:

        x_k     = P_C(y_k - (1/L) grad f(y_k))
        t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
        y_{k+1} = x_k + ((t_k - 1)/t_{k+1}) (x_k - x_{k-1})

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like
        Response of length n.
    lower, upper : float, array-like or None
        Box bounds; ``None`` means unbounded on that side.
    steps : int
        Fixed iteration count.
    lipschitz : float or None
        L; ``None`` uses a fixed 50-step power iteration on X'X.

    Returns
    -------
    RichResult
        ``beta``, ``objective``, ``lipschitz``, ``steps``, ``nactive``,
        ``n``, ``p``.

    References
    ----------
    Beck, A. and Teboulle, M. (2009), SIAM Journal on Imaging Sciences
    2(1), 183-202.  Their FISTA is stated for F = f + g with g closed
    convex and accessed only through its proximal map; taking g as the
    indicator of a convex set turns that map into the projection, which
    is the constrained variant used here.  Standard published form; the
    SIAM article is paywalled and was not read for this implementation.
    """
    Xm = C.mat(X)
    y = C.vec(y)
    steps = int(steps)
    n, p = len(Xm), len(Xm[0])
    if n != len(y):
        raise ValueError("X must have one row per entry of y")
    lo = [-float("inf")] * p if lower is None else _bound(lower, p)
    hi = [float("inf")] * p if upper is None else _bound(upper, p)
    if any(lo[j] > hi[j] for j in range(p)):
        raise ValueError("lower must not exceed upper")
    if lipschitz is None:
        v = [1.0 / math.sqrt(p)] * p
        L = 0.0
        for _ in range(50):
            Xv = [sum(Xm[i][j] * v[j] for j in range(p)) for i in range(n)]
            w = [sum(Xm[i][j] * Xv[i] for i in range(n)) for j in range(p)]
            nw = math.sqrt(sum(t * t for t in w))
            if nw == 0.0:
                L = 0.0
                break
            v = [t / nw for t in w]
            L = nw
    else:
        L = float(lipschitz)
    if L <= 0.0:
        L = 1.0
    x = [min(max(0.0, lo[j]), hi[j]) for j in range(p)]
    yv = list(x)
    t = 1.0
    for _ in range(steps):
        r = [sum(Xm[i][j] * yv[j] for j in range(p)) - y[i] for i in range(n)]
        g = [sum(Xm[i][j] * r[i] for i in range(n)) for j in range(p)]
        xn = [min(max(yv[j] - g[j] / L, lo[j]), hi[j]) for j in range(p)]
        tn = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        w = (t - 1.0) / tn
        yv = [xn[j] + w * (xn[j] - x[j]) for j in range(p)]
        x = xn
        t = tn
    res = [sum(Xm[i][j] * x[j] for j in range(p)) - y[i] for i in range(n)]
    act = sum(1 for j in range(p) if x[j] == lo[j] or x[j] == hi[j])
    return RichResult(payload={
        "beta": x, "objective": 0.5 * sum(e * e for e in res),
        "lipschitz": L, "steps": steps, "nactive": act, "n": n, "p": p,
        "method": "Accelerated projected gradient on a box (Beck-Teboulle 2009)"})


def _bound(b, p):
    try:
        v = C.vec(b)
    except Exception:
        v = [float(b)]
    if len(v) == 1:
        return [v[0]] * p
    if len(v) != p:
        raise ValueError("bound must be scalar or of length p")
    return v


agd_constrained = agdproj


def cheatsheet():
    return "agdcon: Accelerated projected gradient on a box."
