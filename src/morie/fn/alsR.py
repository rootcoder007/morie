# morie.fn -- function file (rootcoder007/morie)
"""Alternating least squares for implicit-feedback matrix factorisation."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["alsmf", "als"]


def alsmf(R, f=2, lam=0.1, alpha=40.0, steps=10, X0=None, Y0=None):
    """Implicit-feedback ALS with per-observation confidence weights.

    Implicit feedback records how often something happened, not how much
    it was liked.  The counts are therefore split into a binary preference
    and a confidence in it:

        p_ui = 1 if r_ui > 0 else 0,      c_ui = 1 + alpha r_ui

    and the factorisation minimises

        sum_{u,i} c_ui (p_ui - x_u' y_i)^2 + lam ( sum_u ||x_u||^2
                                                 + sum_i ||y_i||^2 ),

    which is quadratic in x with y fixed and vice versa, so the exact
    alternating solutions are

        x_u = (Y' C^u Y + lam I)^{-1} Y' C^u p(u)
        y_i = (X' C^i X + lam I)^{-1} X' C^i p(i).

    Parameters
    ----------
    R : array-like, shape (m, n)
        Non-negative observation counts r_ui.
    f : int
        Number of latent factors.
    lam : float
        Ridge penalty.
    alpha : float
        Confidence scaling; the paper's default is 40.
    steps : int
        Fixed number of alternating sweeps.
    X0, Y0 : array-like or None
        Starting factors.  ``None`` uses the deterministic start
        x_uk = ((u + k) mod 5 + 1)/10, y_ik = ((i + 2k) mod 7 + 1)/10,
        identical in every arm.

    Returns
    -------
    RichResult
        ``X``, ``Y``, ``loss``, ``fitted``, ``m``, ``n``, ``f``,
        ``steps``.

    References
    ----------
    Hu, Y., Koren, Y. and Volinsky, C. (2008), "Collaborative filtering
    for implicit feedback datasets", IEEE International Conference on
    Data Mining, 263-272.  Equation (3) is c_ui = 1 + alpha r_ui, the
    cost function is their Sect. 4 objective, and the alternating
    solutions are their Equations (4) and (5).  Read from the authors'
    own PDF at yifanhu.net/PUB/cf.pdf.
    """
    Rm = C.mat(R)
    m, n = len(Rm), len(Rm[0])
    f = int(f)
    if f < 1:
        raise ValueError("f must be at least 1")
    if any(v < 0.0 for r in Rm for v in r):
        raise ValueError("counts must be non-negative")
    lam = float(lam)
    al = float(alpha)
    Cf = [[1.0 + al * Rm[u][i] for i in range(n)] for u in range(m)]
    P = [[1.0 if Rm[u][i] > 0.0 else 0.0 for i in range(n)] for u in range(m)]
    if X0 is None:
        X = [[((u + k) % 5 + 1) / 10.0 for k in range(f)] for u in range(m)]
    else:
        X = [[float(v) for v in r] for r in C.mat(X0)]
    if Y0 is None:
        Y = [[((i + 2 * k) % 7 + 1) / 10.0 for k in range(f)]
             for i in range(n)]
    else:
        Y = [[float(v) for v in r] for r in C.mat(Y0)]
    for _ in range(int(steps)):
        for u in range(m):
            A = [[sum(Cf[u][i] * Y[i][a] * Y[i][b] for i in range(n))
                  + (lam if a == b else 0.0) for b in range(f)]
                 for a in range(f)]
            rhs = [sum(Cf[u][i] * P[u][i] * Y[i][a] for i in range(n))
                   for a in range(f)]
            X[u] = C.solvev(A, rhs)
        for i in range(n):
            A = [[sum(Cf[u][i] * X[u][a] * X[u][b] for u in range(m))
                  + (lam if a == b else 0.0) for b in range(f)]
                 for a in range(f)]
            rhs = [sum(Cf[u][i] * P[u][i] * X[u][a] for u in range(m))
                   for a in range(f)]
            Y[i] = C.solvev(A, rhs)
    fit = [[sum(X[u][a] * Y[i][a] for a in range(f)) for i in range(n)]
           for u in range(m)]
    loss = sum(Cf[u][i] * (P[u][i] - fit[u][i]) ** 2
               for u in range(m) for i in range(n))
    loss += lam * (sum(v * v for r in X for v in r)
                   + sum(v * v for r in Y for v in r))
    return RichResult(payload={
        "X": X, "Y": Y, "loss": loss, "fitted": fit, "m": m, "n": n,
        "f": f, "steps": int(steps),
        "method": "Implicit-feedback ALS (Hu-Koren-Volinsky 2008 eqs. 3-5)"})


als = alsmf


def cheatsheet():
    return "alsR: Alternating least squares for implicit-feedback matrix factorisation."
