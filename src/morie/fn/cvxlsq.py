# morie.fn -- function file (rootcoder007/morie)
"""Least squares -- Boyd & Vandenberghe Sec. 1.2.1 / 4.4.1."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_least_squares"]


def boyd_least_squares(A, b, rcond=None):
    r"""Minimise :math:`\lVert Ax - b\rVert_2^2`.

    The one convex problem with a closed-form solution:
    :math:`A^\top A x = A^\top b`. That is why it is the workhorse and
    also why it is over-used -- an analytic solution is not a reason to
    prefer a model.

    The normal equations are solved here by ``lstsq`` (an SVD), never by
    forming and inverting :math:`A^\top A`, whose condition number is the
    SQUARE of A's. On a mildly ill-conditioned design that squaring is the
    difference between a usable answer and noise.

    Parameters
    ----------
    A : array-like
        Design matrix ``(m, n)``.
    b : array-like
        Right-hand side ``(m,)``.
    rcond : float, optional
        Singular-value cutoff passed to ``lstsq``.

    Returns
    -------
    RichResult
        ``x``, ``residual``, ``rss``, ``rank``, ``condition_number``,
        ``underdetermined``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> r = boyd_least_squares(A, [1.0, 2.0, 3.0])
    >>> [round(float(v), 6) for v in r["x"]]
    [1.0, 2.0]

    Residuals are orthogonal to the column space -- the normal equations
    restated, and the check that the solve actually solved.

    >>> bool(np.max(np.abs(A.T @ r["residual"])) < 1e-10)
    True

    A rank-deficient design is reported rather than silently given the
    minimum-norm solution as if it were unique.

    >>> d = boyd_least_squares([[1.0, 1.0], [2.0, 2.0]], [1.0, 2.0])
    >>> int(d["rank"]), bool(d["underdetermined"])
    (1, True)
    """
    Am = np.atleast_2d(np.asarray(A, dtype=float))
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    if Am.shape[0] != bv.size:
        raise ValueError(f"A has {Am.shape[0]} rows but b has {bv.size}")
    x, _, rank, sv = np.linalg.lstsq(Am, bv, rcond=rcond)
    resid = bv - Am @ x
    cond = float(sv.max() / sv.min()) if sv.size and sv.min() > 0 else float("inf")
    under = bool(rank < Am.shape[1])
    return RichResult(
        title="Least squares",
        summary_lines=[("m", int(Am.shape[0])), ("n", int(Am.shape[1])),
                       ("rss", float(resid @ resid)), ("rank", int(rank))],
        warnings=["A is rank deficient; lstsq returns the minimum-norm "
                  "solution, which is one of infinitely many"] if under else [],
        payload={
            "x": x, "residual": resid, "rss": float(resid @ resid),
            "rank": int(rank), "singular_values": sv,
            "condition_number": cond, "underdetermined": under,
            "method": "boyd_least_squares",
        },
    )


def cheatsheet():
    return "cvxlsq: solve by SVD, never by inverting A'A -- that squares the condition number"
