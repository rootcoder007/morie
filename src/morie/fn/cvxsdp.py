# morie.fn -- function file (rootcoder007/morie)
"""Semidefinite program -- Boyd & Vandenberghe Sec. 4.6.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from ._sdp import solve_sdp

__all__ = ["boyd_sdp"]


def boyd_sdp(c, F, x0=None, tol=1e-09):
    r"""Solve :math:`\min c^{\top}x` s.t.
    :math:`F_0 + \sum_i x_i F_i \succeq 0`.

    The inequality is the LOEWNER order -- the slack matrix must be
    positive semidefinite, not entrywise nonnegative. Confusing the two
    is the standard first mistake with SDPs, and it is not a small one:
    a matrix of all-positive entries is routinely indefinite.

    SDP sits at the top of the tractable hierarchy. An LP is the special
    case where every :math:`F_i` is diagonal (on diagonal matrices the
    Loewner order IS the entrywise order), an SOCP embeds via Schur
    complements, and problems with no LP or QP formulation at all --
    minimising a maximum eigenvalue, relaxing a combinatorial quadratic
    -- fall out immediately.

    Solved by a log-det barrier with Newton centering and outer
    path-following, which is why the reported suboptimality is a
    certified bound :math:`m/t` rather than a guess.

    Parameters
    ----------
    c : array-like
        Objective, length ``n``.
    F : sequence of array-like
        ``n + 1`` symmetric matrices: ``F[0]`` is the constant term
        :math:`F_0`, the rest multiply :math:`x_1 \dots x_n`.
    x0 : array-like, optional
        Strictly feasible starting point. One is searched for if omitted.
    tol : float
        Stop when the barrier gap bound ``m / t`` falls below this.

    Returns
    -------
    RichResult
        ``x``, ``objective``, ``slack`` (the matrix :math:`F(x)`),
        ``eigenvalues``, ``gap_bound``, ``feasible``,
        ``strictly_feasible``, ``active``, ``converged``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.
    Vandenberghe, L., & Boyd, S. (1996). Semidefinite programming.
        *SIAM Review*, 38(1), 49-95.

    Examples
    --------
    Minimising the largest eigenvalue of a matrix is an SDP and nothing
    simpler: ``t*I - A >= 0`` holds exactly when ``t >= lambda_max(A)``,
    so minimising ``t`` under that LMI returns the eigenvalue.

    >>> import numpy as np
    >>> A = np.array([[2.0, 1.0], [1.0, 2.0]])
    >>> r = boyd_sdp([1.0], [-A, np.eye(2)])
    >>> round(float(r["objective"]), 6)
    3.0
    >>> round(float(np.linalg.eigvalsh(A)[-1]), 6)
    3.0

    At the optimum the LMI is TIGHT -- the slack matrix is singular,
    which is the SDP way of saying the constraint is active. A strictly
    feasible optimum would have meant the objective could still fall.

    >>> bool(r["active"]), bool(r["strictly_feasible"])
    (True, False)

    An LP is the diagonal special case. Maximise ``x1 + x2`` subject to
    ``x <= 1`` and ``x >= 0``, written as one 4x4 diagonal LMI.

    >>> F0 = np.diag([1.0, 1.0, 0.0, 0.0])
    >>> F1 = np.diag([-1.0, 0.0, 1.0, 0.0])
    >>> F2 = np.diag([0.0, -1.0, 0.0, 1.0])
    >>> lp = boyd_sdp([-1.0, -1.0], [F0, F1, F2])
    >>> [round(float(v), 5) for v in lp["x"]]
    [1.0, 1.0]

    The barrier bound is a certificate, not an estimate: the true
    optimum is guaranteed to lie within ``gap_bound`` of the value
    reported.

    >>> bool(r["gap_bound"] < 1e-06)
    True

    An empty interior is refused rather than approximated. Slater's
    condition is what the barrier method needs in order to start at all,
    so a constraint set without a strictly feasible point has to say so.

    >>> boyd_sdp([1.0], [np.diag([-1.0, -1.0]), np.zeros((2, 2))])
    Traceback (most recent call last):
        ...
    ValueError: no strictly feasible point found; the constraint set is empty or has empty interior (Slater fails)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float)).ravel()
    mats = [np.atleast_2d(np.asarray(Fi, dtype=float)) for Fi in F]
    if len(mats) != c.size + 1:
        raise ValueError(
            f"c has {c.size} entries, so F needs {c.size + 1} matrices "
            f"(F0 plus one per variable), got {len(mats)}")
    for k, Fi in enumerate(mats):
        if not np.allclose(Fi, Fi.T, atol=1e-10):
            raise ValueError(f"F[{k}] is not symmetric")
    x, info = solve_sdp(c, mats[0], mats[1:], x0=x0, tol=tol)
    if x is None:
        raise ValueError(info["message"])
    ev = info["eigenvalues"]
    scale = max(1.0, float(np.abs(ev).max()))
    return RichResult(
        title="Semidefinite program",
        summary_lines=[("n", int(c.size)), ("block", int(mats[0].shape[0])),
                       ("objective", info["objective"]),
                       ("min eigenvalue", float(ev[0])),
                       ("gap bound", info["gap_bound"])],
        payload={
            "x": x, "objective": info["objective"],
            "slack": info["slack"], "eigenvalues": ev,
            "gap_bound": info["gap_bound"],
            "feasible": bool(ev[0] > -1e-08 * scale),
            "strictly_feasible": bool(ev[0] > 1e-06 * scale),
            "active": bool(ev[0] <= 1e-06 * scale),
            "converged": bool(info["converged"]),
            "method": "boyd_sdp",
        },
    )


def cheatsheet():
    return "cvxsdp: LOEWNER order, not entrywise -- an all-positive matrix is routinely indefinite. LP = diagonal case"
