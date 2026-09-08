# morie.fn -- function file (rootcoder007/morie)
"""Newton-Raphson minimisation."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["newton_raphson"]


def newton_raphson(f, grad_f, hess_f, x0, n_iter=50, tol=1e-12):
    """Newton's method on a smooth multivariate objective.

    Formula: ``x_{t+1} = x_t - H(x_t)^{-1} g(x_t)``.

    The step is obtained by SOLVING ``H d = -g``, never by forming
    ``H^{-1}``: inverting costs three times the work and loses the
    conditioning that the solve keeps.  A general solve is used rather
    than a Cholesky factorisation, because ``H`` is only guaranteed
    positive definite at a minimum, not on the way to one -- on a
    log-likelihood being maximised the caller passes the Hessian of the
    log-likelihood and the method descends on its negative.

    Determinism: a fixed iteration cap with a gradient-norm stopping
    rule; no line search, no random restarts.

    Parameters
    ----------
    f : callable
        Objective, ``f(x) -> float``.
    grad_f : callable
        Gradient, ``grad_f(x) -> length-p sequence``.
    hess_f : callable
        Hessian, ``hess_f(x) -> p by p sequence``.
    x0 : array-like, shape (p,)
        Starting point.
    n_iter : int, default 50
        Maximum iterations.
    tol : float, default 1e-12
        Stop when the Euclidean gradient norm falls below this.

    Returns
    -------
    RichResult
        ``x`` (the solution), ``estimate`` (``f`` at the solution),
        ``fval``, ``grad_norm``, ``iterations``, ``converged``, ``p``.

    References
    ----------
    Newton, I. (1669/1711).  De analysi per aequationes numero
    terminorum infinitas; Raphson, J. (1690).  Analysis aequationum
    universalis.  The modern statement is Nocedal, J. & Wright, S. J.
    (2006), Numerical Optimization, 2nd ed., Springer, algorithm 3.2.
    """
    x = C.vec(x0)
    p = len(x)
    if p == 0:
        raise ValueError("newton_raphson: x0 is empty")
    nit = int(n_iter)
    if nit < 0:
        raise ValueError("newton_raphson: n_iter must be non-negative")
    it = 0
    g = [float(v) for v in grad_f(x)]
    if len(g) != p:
        raise ValueError("newton_raphson: grad_f returned the wrong length")
    gn = math.sqrt(sum(v * v for v in g))
    while it < nit and gn > tol:
        H = C.mat(hess_f(x))
        if len(H) != p or any(len(r) != p for r in H):
            raise ValueError("newton_raphson: hess_f returned the wrong shape")
        d = C.solvev(H, [-v for v in g])
        x = [x[i] + d[i] for i in range(p)]
        it += 1
        g = [float(v) for v in grad_f(x)]
        gn = math.sqrt(sum(v * v for v in g))
    fv = float(f(x))
    return RichResult(payload={
        "x": x, "estimate": fv, "fval": fv, "grad_norm": gn,
        "iterations": it, "converged": 1.0 if gn <= tol else 0.0, "p": p,
        "method": "Newton-Raphson"})


def cheatsheet():
    return "newraf: Newton-Raphson minimisation"


newtonraphson = newton_raphson
