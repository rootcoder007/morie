# morie.fn -- function file (rootcoder007/morie)
"""Basis pursuit denoising -- Boyd & Vandenberghe Sec. 6.3.2."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_basis_pursuit"]


def boyd_basis_pursuit(A, b, eps=0.0):
    r"""Solve :math:`\min \lVert x\rVert_1` s.t.
    :math:`\lVert Ax - b\rVert_2 \le \varepsilon`.

    The constrained form asks the question a practitioner actually has:
    *given that my measurements carry noise of size* :math:`\varepsilon`,
    *what is the sparsest explanation?* Its Lagrangian twin -- the LASSO,
    :math:`\min \lVert Ax-b\rVert_2^2 + \lambda\lVert x\rVert_1` -- traces
    the same path but is parameterised by a :math:`\lambda` with no
    physical meaning, so the two are equivalent only after you have
    found the :math:`\lambda` matching your :math:`\varepsilon`.

    The mechanism is geometric. The :math:`\ell_1` ball has vertices ON
    the coordinate axes, so the first point at which it touches the
    feasible set is generically a point with zero coordinates. The
    :math:`\ell_2` ball is smooth and touches at an interior direction,
    which is why ridge regression shrinks but never zeroes.

    Solved as the epigraph problem
    :math:`\min \sum t_i` s.t. :math:`-t \le x \le t` and
    :math:`\lVert Ax-b\rVert_2^2 \le \varepsilon^2`, a linear objective
    with linear and one quadratic constraint.

    Parameters
    ----------
    A : array-like
        Dictionary, ``(m, n)``; typically ``m < n``.
    b : array-like
        Observations, length ``m``.
    eps : float
        Residual budget. ``0`` demands exact reconstruction.

    Returns
    -------
    RichResult
        ``x``, ``l1_norm``, ``residual``, ``residual_norm``,
        ``n_nonzero``, ``support``, ``feasible``, ``trivial``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.
    Chen, S. S., Donoho, D. L., & Saunders, M. A. (1998). Atomic
        decomposition by basis pursuit. *SIAM Journal on Scientific
        Computing*, 20(1), 33-61.

    Examples
    --------
    Two measurements, three atoms. The third column explains both
    observations at once, so l1 spends 1.25 on it rather than 2.0 spread
    across the first two -- and the answer is genuinely sparse, not
    merely small.

    >>> import numpy as np
    >>> A = np.array([[1.0, 0.0, 0.8], [0.0, 1.0, 0.8]])
    >>> b = np.array([1.0, 1.0])
    >>> r = boyd_basis_pursuit(A, b, eps=0.0)
    >>> int(r["n_nonzero"]), [int(i) for i in r["support"]]
    (1, [2])
    >>> round(float(r["l1_norm"]), 4)
    1.25

    Allowing noise buys a slightly cheaper explanation, and the solver
    spends the ENTIRE budget doing so -- the constraint is active,
    because the objective strictly improves as the residual grows.

    >>> n = boyd_basis_pursuit(A, b, eps=0.01)
    >>> bool(n["l1_norm"] < r["l1_norm"])
    True
    >>> round(float(n["residual_norm"]), 4)
    0.01

    Past ``|b|_2`` the zero vector is itself feasible, so the sparsest
    explanation is no explanation at all. Worth flagging rather than
    returning as a result: an all-zero recovery usually means eps was
    set too loosely, not that the signal was absent.

    >>> z = boyd_basis_pursuit(A, b, eps=2.0)
    >>> int(z["n_nonzero"]), bool(z["trivial"])
    (0, True)
    """
    from scipy.optimize import linprog, minimize

    Am = np.atleast_2d(np.asarray(A, dtype=float))
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    eps = float(eps)
    m, n = Am.shape
    if bv.size != m:
        raise ValueError(f"A has {m} rows but b has {bv.size} entries")
    if eps < 0:
        raise ValueError(f"eps must be nonnegative, got {eps}")
    if not (np.all(np.isfinite(Am)) and np.all(np.isfinite(bv))):
        raise ValueError("A and b must be finite")

    if eps >= np.linalg.norm(bv):
        # x = 0 is feasible and no vector has a smaller l1 norm.
        x = np.zeros(n)
    elif eps == 0.0:
        # Exact reconstruction is an LP, so solve it as one rather than
        # feeding a degenerate equality to a general NLP solver.
        c = np.r_[np.zeros(n), np.ones(n)]
        A_ub = np.block([[np.eye(n), -np.eye(n)], [-np.eye(n), -np.eye(n)]])
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(2 * n),
                      A_eq=np.hstack([Am, np.zeros((m, n))]), b_eq=bv,
                      bounds=[(None, None)] * n + [(0.0, None)] * n,
                      method="highs")
        if res.status != 0:
            raise ValueError(f"exact basis pursuit infeasible: {res.message}")
        x = np.asarray(res.x[:n], dtype=float)
    else:
        x0 = np.linalg.lstsq(Am, bv, rcond=None)[0]
        z0 = np.r_[x0, np.abs(x0) + 1e-06]
        eye = np.eye(n)
        cons = [
            {"type": "ineq",
             "fun": lambda z: z[n:] - z[:n],
             "jac": lambda z: np.hstack([-eye, eye])},
            {"type": "ineq",
             "fun": lambda z: z[n:] + z[:n],
             "jac": lambda z: np.hstack([eye, eye])},
            {"type": "ineq",
             "fun": lambda z: eps**2 - np.sum((Am @ z[:n] - bv) ** 2),
             "jac": lambda z: np.r_[-2.0 * Am.T @ (Am @ z[:n] - bv),
                                    np.zeros(n)]},
        ]
        obj = np.r_[np.zeros(n), np.ones(n)]
        res = minimize(lambda z: float(obj @ z), z0,
                       jac=lambda z: obj, constraints=cons, method="SLSQP",
                       options={"maxiter": 800, "ftol": 1e-12})
        x = np.asarray(res.x[:n], dtype=float)
    # SLSQP lands microscopically off zero on the coordinates it means
    # to kill; without a cut the support count is meaningless.
    x[np.abs(x) < 1e-08] = 0.0
    resid = Am @ x - bv
    rnorm = float(np.linalg.norm(resid))
    return RichResult(
        title="Basis pursuit denoising",
        summary_lines=[("m", int(m)), ("n", int(n)), ("eps", eps),
                       ("l1 norm", float(np.abs(x).sum())),
                       ("residual", rnorm),
                       ("nonzeros", int(np.count_nonzero(x)))],
        payload={
            "x": x, "l1_norm": float(np.abs(x).sum()),
            "residual": resid, "residual_norm": rnorm,
            "n_nonzero": int(np.count_nonzero(x)),
            "support": np.flatnonzero(x),
            "feasible": bool(rnorm <= eps + 1e-06),
            "trivial": bool(np.count_nonzero(x) == 0),
            "eps": eps, "method": "boyd_basis_pursuit",
        },
    )


def cheatsheet():
    return "cvxbgm: l1 ball has vertices ON the axes -- that is the whole reason it gives sparsity, l2 cannot"
