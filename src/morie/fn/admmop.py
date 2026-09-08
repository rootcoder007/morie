# morie.fn -- function file (rootcoder007/morie)
"""ADMM in scaled form for the LASSO."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["admmlasso", "admm"]


def _soft(v, t):
    if v > t:
        return v - t
    if v < -t:
        return v + t
    return 0.0


def admmlasso(X, y, lam, rho=1.0, steps=100):
    """Alternating direction method of multipliers for the LASSO.

    Splitting minimise 0.5||Xx - y||^2 + lam||z||_1 subject to x - z = 0
    gives the augmented Lagrangian L_rho = 0.5||Xx-y||^2 + lam||z||_1 +
    (rho/2)||x - z + u||^2 in scaled dual form, and the three updates

        x^{k+1} = (X'X + rho I)^{-1} (X'y + rho (z^k - u^k))
        z^{k+1} = S_{lam/rho}(x^{k+1} + u^k)
        u^{k+1} = u^k + x^{k+1} - z^{k+1}

    with S the soft-thresholding operator.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like
        Response of length n.
    lam : float
        L1 penalty, non-negative.
    rho : float
        Augmented-Lagrangian parameter, strictly positive.
    steps : int
        Fixed iteration count; no tolerance early exit.

    Returns
    -------
    RichResult
        ``x``, ``z``, ``u``, ``objective``, ``primalres``, ``dualres``,
        ``rho``, ``steps``, ``n``, ``p``.

    References
    ----------
    Boyd, S., Parikh, N., Chu, E., Peleato, B. and Eckstein, J. (2011),
    "Distributed optimization and statistical learning via the alternating
    direction method of multipliers", Foundations and Trends in Machine
    Learning 3(1), 1-122.  Section 6.4 gives exactly these three LASSO
    updates; Sect. 3.1.1 gives the scaled dual form.  Standard published
    form of ADMM; the monograph was not in the local corpus and was not
    read for this implementation.
    """
    Xm = C.mat(X)
    y = C.vec(y)
    lam = float(lam)
    rho = float(rho)
    steps = int(steps)
    n, p = len(Xm), len(Xm[0])
    if n != len(y):
        raise ValueError("X must have one row per entry of y")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    if rho <= 0.0:
        raise ValueError("rho must be strictly positive")
    A = [[sum(Xm[i][a] * Xm[i][b] for i in range(n)) + (rho if a == b else 0.0)
          for b in range(p)] for a in range(p)]
    Xty = [sum(Xm[i][a] * y[i] for i in range(n)) for a in range(p)]
    x = [0.0] * p
    z = [0.0] * p
    u = [0.0] * p
    dual = 0.0
    for _ in range(steps):
        rhs = [Xty[a] + rho * (z[a] - u[a]) for a in range(p)]
        x = C.solvev(A, rhs)
        zold = z
        z = [_soft(x[a] + u[a], lam / rho) for a in range(p)]
        u = [u[a] + x[a] - z[a] for a in range(p)]
        dual = rho * C.norm2([z[a] - zold[a] for a in range(p)])
    res = [sum(Xm[i][a] * z[a] for a in range(p)) - y[i] for i in range(n)]
    obj = 0.5 * sum(e * e for e in res) + lam * sum(abs(v) for v in z)
    return RichResult(payload={
        "x": x, "z": z, "u": u, "objective": obj,
        "primalres": C.norm2([x[a] - z[a] for a in range(p)]),
        "dualres": dual, "rho": rho, "steps": steps, "n": n, "p": p,
        "method": "ADMM for the LASSO, scaled form (Boyd et al. 2011 Sect. 6.4)"})


admm = admmlasso


def cheatsheet():
    return "admmop: ADMM in scaled form for the LASSO."
