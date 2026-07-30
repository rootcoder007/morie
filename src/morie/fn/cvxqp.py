# morie.fn -- function file (rootcoder007/morie)
"""Quadratic program -- Boyd & Vandenberghe Sec. 4.4."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_quadratic_program"]


def boyd_quadratic_program(P, q, G=None, h=None, A=None, b=None,
                           max_iter=200, tol=1e-10):
    r"""Solve :math:`\min \tfrac12 x^\top Px + q^\top x` subject to
    :math:`Gx \le h` and :math:`Ax = b`.

    Convex only when P is positive SEMI-definite, and that is checked
    rather than assumed: with an indefinite P the problem is NP-hard in
    general and any answer a convex solver returns is a local point
    presented as a global one.

    With equality constraints alone the solution is the KKT linear system

    .. math::
        \begin{bmatrix} P & A^\top \\ A & 0 \end{bmatrix}
        \begin{bmatrix} x \\ \nu \end{bmatrix}
        = \begin{bmatrix} -q \\ b \end{bmatrix},

    solved directly. Inequalities are handled by an active-set loop over
    that same system.

    Parameters
    ----------
    P : array-like
        Quadratic term, symmetric positive semi-definite.
    q : array-like
        Linear term.
    G, h : array-like, optional
        Inequality constraints.
    A, b : array-like, optional
        Equality constraints.
    max_iter, tol
        Active-set controls.

    Returns
    -------
    RichResult
        ``x``, ``value``, ``nu`` (equality multipliers), ``lambda``
        (inequality multipliers), ``active_set``, ``psd``, ``converged``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Unconstrained, the solution is the linear solve -P^-1 q.

    >>> import numpy as np
    >>> P = np.array([[2.0, 0.0], [0.0, 4.0]])
    >>> r = boyd_quadratic_program(P, [-2.0, -8.0])
    >>> [round(float(v), 6) for v in r["x"]]
    [1.0, 2.0]

    An equality constraint is met exactly, through the KKT system.

    >>> e = boyd_quadratic_program(P, [-2.0, -8.0], A=[[1.0, 1.0]], b=[1.0])
    >>> round(float(e["x"].sum()), 8)
    1.0

    An active inequality is tight and carries a non-negative multiplier;
    complementary slackness holds.

    >>> g = boyd_quadratic_program(P, [-2.0, -8.0], G=[[1.0, 1.0]], h=[1.0])
    >>> bool(abs(g["x"].sum() - 1.0) < 1e-8 and g["lambda"][0] >= -1e-9)
    True

    An indefinite P is refused rather than solved as if it were convex.

    >>> boyd_quadratic_program([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0])
    Traceback (most recent call last):
        ...
    ValueError: P must be positive semi-definite for a convex QP; its smallest eigenvalue is -1
    """
    Pm = np.atleast_2d(np.asarray(P, dtype=float))
    qv = np.atleast_1d(np.asarray(q, dtype=float)).ravel()
    n = qv.size
    if Pm.shape != (n, n):
        raise ValueError(f"P must be ({n}, {n}) to match q")
    Pm = 0.5 * (Pm + Pm.T)
    w = np.linalg.eigvalsh(Pm)
    if w.min() < -1e-08 * max(1.0, abs(w).max()):
        raise ValueError(
            "P must be positive semi-definite for a convex QP; its "
            f"smallest eigenvalue is {w.min():g}")
    Am = np.atleast_2d(np.asarray(A, dtype=float)) if A is not None else np.zeros((0, n))
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel() if b is not None else np.zeros(0)
    Gm = np.atleast_2d(np.asarray(G, dtype=float)) if G is not None else np.zeros((0, n))
    hv = np.atleast_1d(np.asarray(h, dtype=float)).ravel() if h is not None else np.zeros(0)

    def kkt(active):
        C = np.vstack([Am, Gm[active]]) if active.size or Am.shape[0] else Am
        d = np.r_[bv, hv[active]]
        k = C.shape[0]
        K = np.block([[Pm, C.T], [C, np.zeros((k, k))]]) if k else Pm
        rhs = np.r_[-qv, d] if k else -qv
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        return sol[:n], sol[n:]

    active = np.zeros(0, dtype=int)
    conv = False
    for it in range(int(max_iter)):
        x, mult = kkt(active)
        viol = Gm @ x - hv if Gm.shape[0] else np.zeros(0)
        worst = int(np.argmax(viol)) if viol.size else -1
        if viol.size and viol[worst] > tol and worst not in active:
            active = np.r_[active, worst]
            continue
        # Drop any active constraint whose multiplier has gone negative:
        # it is pulling the wrong way and does not belong in the set.
        lam = mult[Am.shape[0]:] if active.size else np.zeros(0)
        if lam.size and lam.min() < -tol:
            active = np.delete(active, int(np.argmin(lam)))
            continue
        conv = True
        break
    x, mult = kkt(active)
    nu = mult[:Am.shape[0]] if Am.shape[0] else np.zeros(0)
    lam_full = np.zeros(Gm.shape[0])
    if active.size:
        lam_full[active] = mult[Am.shape[0]:]
    return RichResult(
        title="Quadratic program",
        summary_lines=[("n", int(n)), ("active", int(active.size)),
                       ("value", float(0.5 * x @ Pm @ x + qv @ x)),
                       ("converged", conv)],
        warnings=[] if conv else ["the active-set loop did not settle"],
        payload={
            "x": x, "value": float(0.5 * x @ Pm @ x + qv @ x),
            "nu": nu, "lambda": lam_full, "active_set": active,
            "psd": True, "min_eigenvalue": float(w.min()),
            "converged": conv, "method": "boyd_quadratic_program",
        },
    )


def cheatsheet():
    return "cvxqp: convex only if P is PSD -- indefinite P is NP-hard, so it is refused, not solved"
