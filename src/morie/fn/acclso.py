# morie.fn -- function file (rootcoder007/morie)
"""FISTA for the LASSO."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["fistalasso", "accelerated_lasso"]


def _softthr(v, t):
    if v > t:
        return v - t
    if v < -t:
        return v + t
    return 0.0


def _speclip(Xm, n, p, iters=50):
    """Largest eigenvalue of X'X by a fixed-count power iteration."""
    v = [1.0 / math.sqrt(p)] * p
    lam = 0.0
    for _ in range(iters):
        Xv = [sum(Xm[i][j] * v[j] for j in range(p)) for i in range(n)]
        w = [sum(Xm[i][j] * Xv[i] for i in range(n)) for j in range(p)]
        nw = math.sqrt(sum(t * t for t in w))
        if nw == 0.0:
            return 0.0
        v = [t / nw for t in w]
        lam = nw
    return lam


def fistalasso(X, y, lam, steps=100, lipschitz=None):
    """Fast iterative shrinkage-thresholding for the LASSO.

    Minimises F(b) = 0.5 ||X b - y||_2^2 + lam ||b||_1 by the accelerated
    proximal-gradient scheme of Beck and Teboulle.  With f(b) =
    0.5||Xb-y||^2 the Lipschitz constant of grad f is L = lambda_max(X'X),
    and the proximal map of lam||.||_1 at step 1/L is soft-thresholding at
    lam/L.

    Formula (Beck-Teboulle 2009, Sect. 4, algorithm FISTA with constant
    step size):
        x_k = p_L(y_k) = soft(y_k - (1/L) grad f(y_k), lam/L)
        t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
        y_{k+1} = x_k + ((t_k - 1) / t_{k+1}) (x_k - x_{k-1})
    started from y_1 = x_0 = 0, t_1 = 1.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix, one record per row.
    y : array-like
        Response vector of length n.
    lam : float
        L1 penalty, non-negative.
    steps : int
        Fixed number of FISTA iterations (no tolerance early exit, so the
        result is bit-reproducible).
    lipschitz : float or None
        Step constant L.  ``None`` uses a fixed 50-step power iteration on
        X'X, which is deterministic and identical in every arm.

    Returns
    -------
    RichResult
        ``beta``, ``objective``, ``rss``, ``l1``, ``lipschitz``, ``steps``,
        ``nonzero``, ``n``, ``p``.

    References
    ----------
    Beck, A. and Teboulle, M. (2009), "A fast iterative shrinkage-
    thresholding algorithm for linear inverse problems", SIAM Journal on
    Imaging Sciences 2(1), 183-202.  Section 4 states the constant-step
    FISTA recursion reproduced above; the momentum weight (t_k-1)/t_{k+1}
    and the update t_{k+1}=(1+sqrt(1+4 t_k^2))/2 are theirs.  Standard
    published form of FISTA; the SIAM article itself is paywalled and was
    not read for this implementation.
    """
    Xm = C.mat(X)
    y = C.vec(y)
    lam = float(lam)
    steps = int(steps)
    n, p = len(Xm), len(Xm[0])
    if n != len(y):
        raise ValueError("X must have one row per entry of y")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    L = _speclip(Xm, n, p) if lipschitz is None else float(lipschitz)
    if L <= 0.0:
        L = 1.0
    x = [0.0] * p
    yv = [0.0] * p
    t = 1.0
    for _ in range(steps):
        r = [sum(Xm[i][j] * yv[j] for j in range(p)) - y[i] for i in range(n)]
        g = [sum(Xm[i][j] * r[i] for i in range(n)) for j in range(p)]
        xn = [_softthr(yv[j] - g[j] / L, lam / L) for j in range(p)]
        tn = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        w = (t - 1.0) / tn
        yv = [xn[j] + w * (xn[j] - x[j]) for j in range(p)]
        x = xn
        t = tn
    res = [sum(Xm[i][j] * x[j] for j in range(p)) - y[i] for i in range(n)]
    rss = 0.5 * sum(e * e for e in res)
    l1 = sum(abs(b) for b in x)
    return RichResult(payload={
        "beta": x, "objective": rss + lam * l1, "rss": rss, "l1": l1,
        "lipschitz": L, "steps": steps,
        "nonzero": sum(1 for b in x if b != 0.0), "n": n, "p": p,
        "method": "FISTA for the LASSO (Beck-Teboulle 2009 Sect. 4)"})


accelerated_lasso = fistalasso


def cheatsheet():
    return "acclso: FISTA for the LASSO."
