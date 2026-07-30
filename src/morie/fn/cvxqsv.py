# morie.fn -- function file (rootcoder007/morie)
"""SDP relaxation of a QCQP -- Boyd & Vandenberghe Sec. 5.3.2 / App. B."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from ._sdp import solve_sdp

__all__ = ["boyd_qcqp_relaxation"]


def boyd_qcqp_relaxation(P0, q0, P=(), q=(), r=(), rank_tol=1e-05,
                         tol=1e-09):
    r"""Lagrangian / SDP relaxation of a quadratically constrained QP.

    Every quadratic form in the problem is :math:`\tfrac12\langle P,
    xx^{\top}\rangle` plus a linear term, so substituting a new variable
    :math:`X = xx^{\top}` makes objective and constraints LINEAR in
    :math:`(X, x)`. The whole difficulty of the QCQP is then squeezed
    into the single condition :math:`X = xx^{\top}`, equivalently
    :math:`X \succeq xx^{\top}` together with
    :math:`\operatorname{rank} X = 1`. Keep the first, drop the second,
    and what remains is an SDP:

    .. math::

        \begin{bmatrix} X & x \\ x^{\top} & 1 \end{bmatrix} \succeq 0 .

    Because a relaxation only enlarges the feasible set, its value is
    always a LOWER BOUND on the QCQP -- valid whether or not the
    original problem was convex, which is what makes it the standard
    tool for nonconvex quadratics. And the bound comes with its own
    quality check: if the returned :math:`X` has rank one, then
    :math:`X = xx^{\top}` after all, nothing was relaxed, and
    :math:`x` solves the original problem exactly.

    This is the Goemans-Williamson construction. For MAX-CUT its bound
    can exceed the integral optimum -- the relaxation genuinely loses
    something, and the rank tells you when.

    Parameters
    ----------
    P0 : array-like
        Objective Hessian, ``(n, n)``.
    q0 : array-like
        Objective linear term, length ``n``.
    P : sequence of array-like
        One ``(n, n)`` matrix per constraint. Need NOT be PSD.
    q : sequence of array-like
        One length-``n`` vector per constraint.
    r : array-like
        One scalar per constraint.
    rank_tol : float
        Eigenvalues of ``X`` below this fraction of the largest count as
        zero when reporting the rank.
    tol : float
        Barrier gap tolerance passed to the SDP solver.

    Returns
    -------
    RichResult
        ``X``, ``x``, ``lower_bound``, ``rank``, ``eigenvalues``,
        ``tight`` (rank one, so the bound is attained), ``residual``
        (:math:`\lVert X - xx^{\top}\rVert`), ``gap_bound``,
        ``converged``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.
    Goemans, M. X., & Williamson, D. P. (1995). Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming. *Journal of the ACM*, 42(6),
        1115-1145.

    Examples
    --------
    On a CONVEX QCQP the relaxation gives nothing away. Minimise
    ``|x|^2/2 - 2*x1`` over the unit disc: the bound equals the true
    optimum and the lifted matrix comes back rank one.

    >>> import numpy as np
    >>> r1 = boyd_qcqp_relaxation(np.eye(2), [-2.0, 0.0], P=[np.eye(2)],
    ...                           q=[[0.0, 0.0]], r=[-0.5])
    >>> round(float(r1["lower_bound"]), 4)
    -1.5
    >>> int(r1["rank"]), bool(r1["tight"])
    (1, True)
    >>> [round(float(v), 4) for v in r1["x"]]
    [1.0, 0.0]

    Now the case the method exists for. Minimise
    ``x1x2 + x1x3 + x2x3`` over the cube ``x_i^2 <= 1`` -- the objective
    is multilinear, so its true minimum sits at a vertex and equals -1
    (one variable can disagree with the other two, but not all three
    pairs can disagree at once).

    >>> J = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    >>> box = [2.0 * np.outer(e, e) for e in np.eye(3)]
    >>> g = boyd_qcqp_relaxation(J, np.zeros(3), P=box,
    ...                          q=[np.zeros(3)] * 3, r=[-1.0] * 3)
    >>> round(float(g["lower_bound"]), 3)
    -1.5

    The bound is -3/2, strictly below the true -1: the relaxation has
    lost something real, and it says so through the rank. There is no
    rank-one X achieving -3/2, so no x reproduces the bound.

    >>> int(g["rank"]), bool(g["tight"])
    (2, False)
    >>> bool(g["lower_bound"] < -1.0)
    True

    That gap is not a defect of the solver; it is the frustrated
    triangle. Three variables cannot pairwise disagree, but three
    directions in the PLANE can sit at 120 degrees and do exactly that
    -- which is the rank-2 X the relaxation returns, and the reason
    Goemans-Williamson rounds rather than reads off.

    >>> [round(float(v), 3) for v in np.linalg.eigvalsh(g["X"])]
    [0.0, 1.5, 1.5]
    """
    P0m = np.atleast_2d(np.asarray(P0, dtype=float))
    q0v = np.atleast_1d(np.asarray(q0, dtype=float)).ravel()
    n = q0v.size
    if P0m.shape != (n, n):
        raise ValueError(f"P0 has shape {P0m.shape}, expected ({n}, {n})")
    P0m = 0.5 * (P0m + P0m.T)
    Ps = []
    for Pi in P:
        M = np.atleast_2d(np.asarray(Pi, dtype=float))
        if M.shape != (n, n):
            raise ValueError(f"a P has shape {M.shape}, expected ({n}, {n})")
        Ps.append(0.5 * (M + M.T))
    qs = [np.atleast_1d(np.asarray(qi, dtype=float)).ravel() for qi in q]
    rs = np.atleast_1d(np.asarray(r, dtype=float)).ravel() if Ps else np.zeros(0)
    m = len(Ps)
    if not (len(qs) == rs.size == m):
        raise ValueError(
            f"P, q, r must have the same length; got {m}, {len(qs)}, {rs.size}")
    for i in range(m):
        if qs[i].size != n:
            raise ValueError(f"q[{i}] has {qs[i].size} entries, expected {n}")

    tri = [(i, j) for i in range(n) for j in range(i, n)]
    nv = len(tri) + n
    size = (n + 1) + m

    def blank():
        return np.zeros((size, size))

    # F0: the lifted matrix's constant corner, and the constraints'
    # constant terms on the diagonal block.
    F0 = blank()
    F0[n, n] = 1.0
    for k in range(m):
        F0[n + 1 + k, n + 1 + k] = -float(rs[k])

    Fs, c = [], np.zeros(nv)
    for v, (i, j) in enumerate(tri):
        M = blank()
        M[i, j] = M[j, i] = 1.0
        # <P, X> counts an off-diagonal entry TWICE, since X_ij and X_ji
        # are the same variable; the diagonal only once. Getting this
        # wrong halves or doubles every off-diagonal coefficient and
        # quietly returns a bound for a different problem.
        w = 1.0 if i < j else 0.5
        c[v] = w * P0m[i, j]
        for k in range(m):
            M[n + 1 + k, n + 1 + k] = -w * Ps[k][i, j]
        Fs.append(M)
    for jj in range(n):
        v = len(tri) + jj
        M = blank()
        M[jj, n] = M[n, jj] = 1.0
        c[v] = q0v[jj]
        for k in range(m):
            M[n + 1 + k, n + 1 + k] = -qs[k][jj]
        Fs.append(M)

    # A strictly feasible start: X = alpha*I, x = 0. The lifted matrix
    # is then diagonal and positive for any alpha > 0, so only the
    # constraint slacks need checking.
    start = None
    for alpha in (1e-03, 1e-02, 0.1, 0.5, 1.0, 10.0):
        v0 = np.zeros(nv)
        for v, (i, j) in enumerate(tri):
            if i == j:
                v0[v] = alpha
        vals = [0.5 * alpha * float(np.trace(Ps[k])) + float(rs[k])
                for k in range(m)]
        if all(val < -1e-09 for val in vals):
            start = v0
            break

    sol, info = solve_sdp(c, F0, Fs, x0=start, tol=tol)
    if sol is None:
        raise ValueError(info["message"])
    X = np.zeros((n, n))
    for v, (i, j) in enumerate(tri):
        X[i, j] = X[j, i] = sol[v]
    x = sol[len(tri):].copy()
    ev = np.linalg.eigvalsh(X)
    top = float(ev[-1]) if ev.size else 0.0
    rank = int(np.sum(ev > rank_tol * max(top, 1e-12)))
    resid = float(np.linalg.norm(X - np.outer(x, x)))
    return RichResult(
        title="SDP relaxation of QCQP",
        summary_lines=[("n", int(n)), ("constraints", int(m)),
                       ("lower bound", info["objective"]),
                       ("rank of X", rank),
                       ("tight", bool(rank <= 1))],
        payload={
            "X": X, "x": x, "lower_bound": info["objective"],
            "rank": rank, "eigenvalues": ev,
            "tight": bool(rank <= 1),
            "residual": resid,
            "gap_bound": info["gap_bound"],
            "converged": bool(info["converged"]),
            "method": "boyd_qcqp_relaxation",
        },
    )


def cheatsheet():
    return "cvxqsv: always a LOWER bound, convex or not; rank(X) == 1 means nothing was lost and x is the true answer"
