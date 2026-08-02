# morie.fn -- function file (rootcoder007/morie)
"""QP dual -- Boyd & Vandenberghe Sec. 5.2.4."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_qp_dual"]


def boyd_qp_dual(P, q, G, h):
    r"""Lagrange dual of the inequality-constrained QP.

    For :math:`\min \tfrac12 x^{\top}Px + q^{\top}x` subject to
    :math:`Gx \preceq h` with :math:`P \succ 0`, stationarity gives
    :math:`x = -P^{-1}(q + G^{\top}\lambda)`, and substituting back:

    .. math::

        g(\lambda) = -\tfrac12 \lambda^{\top}GP^{-1}G^{\top}\lambda
        - (h + GP^{-1}q)^{\top}\lambda - \tfrac12 q^{\top}P^{-1}q .

    The cross term :math:`GP^{-1}q` is easy to lose when the formula is
    quoted for the special case :math:`q = 0`; it is carried here.

    The dual is itself a QP, in as many variables as the primal had
    CONSTRAINTS. That is the whole point when constraints are few and
    variables are many, and it is why the SVM is solved in its dual.
    Strong duality holds whenever the primal is feasible -- affine
    constraints satisfy Slater's condition without needing a strictly
    interior point.

    Parameters
    ----------
    P : array-like
        Positive definite Hessian, ``(n, n)``.
    q : array-like
        Linear term, length ``n``.
    G, h : array-like
        Inequality constraints ``Gx <= h``: ``(m, n)`` and length ``m``.

    Returns
    -------
    RichResult
        ``lambda_``, ``dual_value``, ``x``, ``primal_value``, ``gap``,
        ``strong_duality``, ``slack``, ``active``,
        ``complementary_slackness``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Minimise ``|x|^2/2 - x1 - x2`` subject to ``x1 + x2 <= 1``. The
    unconstrained minimiser is (1, 1), which violates the constraint, so
    the solution sits on the boundary at (1/2, 1/2).

    >>> import numpy as np
    >>> P = np.eye(2)
    >>> r = boyd_qp_dual(P, [-1.0, -1.0], [[1.0, 1.0]], [1.0])
    >>> [round(float(v), 6) for v in r["x"]]
    [0.5, 0.5]
    >>> round(float(r["primal_value"]), 6)
    -0.75

    The dual attains the same value -- no gap, as promised for a
    feasible QP with affine constraints.

    >>> round(float(r["dual_value"]), 6)
    -0.75
    >>> bool(r["strong_duality"])
    True

    The multiplier is the shadow price of the binding constraint,
    positive exactly because the constraint bites.

    >>> round(float(r["lambda_"][0]), 6)
    0.5
    >>> [bool(a) for a in r["active"]]
    [True]

    Slacken the constraint past the unconstrained optimum and the
    multiplier drops to zero -- complementary slackness in one line.

    >>> loose = boyd_qp_dual(P, [-1.0, -1.0], [[1.0, 1.0]], [5.0])
    >>> round(float(loose["lambda_"][0]), 9)
    0.0
    >>> [round(float(v), 6) for v in loose["x"]]
    [1.0, 1.0]
    >>> bool(loose["complementary_slackness"])
    True

    An indefinite P is refused rather than pseudo-solved: the dual would
    be unbounded and the returned numbers would mean nothing.

    >>> boyd_qp_dual(np.diag([1.0, -1.0]), [0.0, 0.0], [[1.0, 1.0]], [1.0])
    Traceback (most recent call last):
        ...
    ValueError: P must be positive definite; smallest eigenvalue -1
    """
    from scipy.optimize import minimize

    Pm = np.atleast_2d(np.asarray(P, dtype=float))
    qv = np.atleast_1d(np.asarray(q, dtype=float)).ravel()
    Gm = np.atleast_2d(np.asarray(G, dtype=float))
    hv = np.atleast_1d(np.asarray(h, dtype=float)).ravel()
    n = Pm.shape[0]
    if Pm.shape[0] != Pm.shape[1]:
        raise ValueError("P must be square")
    if qv.size != n:
        raise ValueError(f"P is {n}x{n} but q has {qv.size} entries")
    if Gm.shape[1] != n:
        raise ValueError(f"G has {Gm.shape[1]} columns but P is {n}x{n}")
    if hv.size != Gm.shape[0]:
        raise ValueError(f"G has {Gm.shape[0]} rows but h has {hv.size}")
    Pm = 0.5 * (Pm + Pm.T)
    lo = float(np.linalg.eigvalsh(Pm)[0])
    if lo <= 0:
        # A P that is indefinite (or singular along a feasible
        # direction) leaves the dual unbounded below; refusing is
        # honest, silently pseudo-inverting is not.
        raise ValueError(
            f"P must be positive definite; smallest eigenvalue {lo:.6g}")
    Pinv_q = np.linalg.solve(Pm, qv)
    H = Gm @ np.linalg.solve(Pm, Gm.T)
    H = 0.5 * (H + H.T)
    lin = hv + Gm @ Pinv_q
    const = -0.5 * float(qv @ Pinv_q)

    def neg_g(lam):
        return 0.5 * lam @ H @ lam + lin @ lam - const

    m = Gm.shape[0]
    res = minimize(neg_g, np.zeros(m), jac=lambda lam: H @ lam + lin,
                   method="L-BFGS-B", bounds=[(0.0, None)] * m,
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000})
    lam = np.maximum(np.asarray(res.x, dtype=float), 0.0)
    dual_val = -float(neg_g(lam))
    x = -np.linalg.solve(Pm, qv + Gm.T @ lam)
    primal_val = float(0.5 * x @ Pm @ x + qv @ x)
    slack = hv - Gm @ x
    gap = primal_val - dual_val
    scale = max(1.0, abs(primal_val))
    return RichResult(
        title="QP dual",
        summary_lines=[("n", int(n)), ("m", int(m)),
                       ("primal", primal_val), ("dual", dual_val),
                       ("gap", gap),
                       ("active", int(np.sum(np.abs(slack) < 1e-07)))],
        payload={
            "lambda_": lam, "dual_value": dual_val, "x": x,
            "primal_value": primal_val, "gap": gap,
            "strong_duality": bool(abs(gap) < 1e-06 * scale),
            "slack": slack,
            "active": np.abs(slack) < 1e-07,
            "complementary_slackness": bool(
                np.max(np.abs(lam * slack)) < 1e-06 * scale)
            if m else True,
            "converged": bool(res.success), "method": "boyd_qp_dual",
        },
    )


def cheatsheet():
    return "cvxqpdl: dual QP has one variable per CONSTRAINT -- why the SVM is solved in its dual"
