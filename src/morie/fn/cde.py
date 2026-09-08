# morie.fn -- function file (rootcoder007/morie)
"""Controlled direct effect of a treatment at a fixed mediator value.

Robins, J.M. & Greenland, S. (1992).  Identifiability and
exchangeability for direct and indirect effects.  Epidemiology
3:143-155.

The controlled direct effect contrasts the two treatment arms with
the mediator held at one value for everybody,

    CDE(m) = E[Y(1, m)] - E[Y(0, m)],

which is what distinguishes it from the natural direct effect: the
mediator is set, not left at whatever it would have been.  Under a
linear outcome model with a treatment-by-mediator interaction,

    E[Y | X, M] = b0 + bX X + bM M + bXM X M,

the contrast is bX + bXM m, so it depends on m exactly when the
interaction is present.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["controlled_direct_effect"]


def _ols(Xd, y):
    """Least squares by Gaussian elimination on the normal equations.
    Small designs only -- this is a four-column model."""
    n = len(y)
    p = len(Xd[0])
    A = [[sum(Xd[i][r] * Xd[i][c] for i in range(n)) for c in range(p)]
         + [sum(Xd[i][r] * y[i] for i in range(n))] for r in range(p)]
    for k in range(p):
        piv = max(range(k, p), key=lambda r: abs(A[r][k]))
        if abs(A[piv][k]) < 1e-12:
            raise ValueError("design is rank deficient")
        A[k], A[piv] = A[piv], A[k]
        d = A[k][k]
        A[k] = [v / d for v in A[k]]
        for r in range(p):
            if r != k and A[r][k] != 0.0:
                f = A[r][k]
                A[r] = [A[r][c] - f * A[k][c] for c in range(p + 1)]
    return [A[r][p] for r in range(p)]


def controlled_direct_effect(Y, X, M, m):
    """CDE at mediator value ``m``, from a linear outcome model with a
    treatment-by-mediator interaction.

    Parameters
    ----------
    Y : (n,) outcomes.
    X : (n,) treatment, 0/1.
    M : (n,) mediator.
    m : float, the value the mediator is set to.

    Anchor.  With no interaction the CDE does not depend on m and
    equals the treatment coefficient, which is the total effect under
    this model; ``interaction`` is returned so that degeneracy is
    visible rather than assumed.

    Returns
    -------
    RichResult with keys estimate (the CDE at m), cde, intercept,
    beta_x, beta_m, interaction, se, m, n, method.
    """
    ys = [float(v) for v in Y]
    xs = [float(v) for v in X]
    ms = [float(v) for v in M]
    n = len(ys)
    if not (len(xs) == len(ms) == n):
        raise ValueError("Y, X and M must have the same length")
    if n < 5:
        raise ValueError("need at least five observations for four parameters")
    mv = float(m)
    D = [[1.0, xs[i], ms[i], xs[i] * ms[i]] for i in range(n)]
    b = _ols(D, ys)
    cde_val = b[1] + b[3] * mv
    resid = [ys[i] - sum(D[i][j] * b[j] for j in range(4)) for i in range(n)]
    s2 = sum(r * r for r in resid) / (n - 4)
    # var(bX + m bXM) needs the two diagonal entries and their covariance
    XtX = [[sum(D[i][r] * D[i][c] for i in range(n)) for c in range(4)]
           for r in range(4)]
    inv = _inv4(XtX)
    var = s2 * (inv[1][1] + mv * mv * inv[3][3] + 2.0 * mv * inv[1][3])
    return with_describe_pointer(RichResult(payload={
        "estimate": float(cde_val), "cde": float(cde_val),
        "intercept": float(b[0]), "beta_x": float(b[1]),
        "beta_m": float(b[2]), "interaction": float(b[3]),
        "se": float(math.sqrt(var)) if var > 0 else 0.0,
        "m": mv, "n": n,
        "method": "controlled direct effect (Robins & Greenland 1992)",
    }), "cde")


def _inv4(A):
    """Gauss-Jordan inverse of a small symmetric matrix."""
    p = len(A)
    M = [list(A[r]) + [1.0 if c == r else 0.0 for c in range(p)]
         for r in range(p)]
    for k in range(p):
        piv = max(range(k, p), key=lambda r: abs(M[r][k]))
        if abs(M[piv][k]) < 1e-12:
            raise ValueError("design is rank deficient")
        M[k], M[piv] = M[piv], M[k]
        d = M[k][k]
        M[k] = [v / d for v in M[k]]
        for r in range(p):
            if r != k and M[r][k] != 0.0:
                f = M[r][k]
                M[r] = [M[r][c] - f * M[k][c] for c in range(2 * p)]
    return [row[p:] for row in M]


def cheatsheet():
    return "cde: Controlled direct effect"


# compact alias per ledger/NAMING.md
ctrldirect = controlled_direct_effect
