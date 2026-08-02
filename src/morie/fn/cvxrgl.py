# morie.fn -- function file (rootcoder007/morie)
"""Regularized least squares -- Boyd & Vandenberghe Sec. 6.3.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_regularized_ls"]


def boyd_regularized_ls(A, b, delta=1.0):
    r"""Minimise :math:`\lVert Ax - b\rVert_2^2 + \delta\lVert x\rVert_2^2`.

    The penalty does two separate jobs that are easy to conflate. It trades
    fit against solution size, which is the bias-variance story; and it
    makes :math:`A^\top A + \delta I` strictly positive definite, which
    means the problem has a UNIQUE solution even when A is rank deficient.
    The second is not a statistical argument at all -- it is why ridge
    works on a singular design where plain least squares has no unique
    answer.

    Solved through the stacked system :math:`[A; \sqrt\delta I]`, so the
    conditioning stays that of the augmented matrix rather than of
    :math:`A^\top A`.

    Parameters
    ----------
    A, b : array-like
        Design and right-hand side.
    delta : float
        Regularisation weight, non-negative.

    Returns
    -------
    RichResult
        ``x``, ``rss``, ``penalty``, ``objective``, ``effective_df``,
        ``shrinkage``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A singular design has no unique least-squares solution but a perfectly
    unique ridge solution.

    >>> import numpy as np
    >>> A = np.array([[1.0, 1.0], [1.0, 1.0]])
    >>> r = boyd_regularized_ls(A, [2.0, 2.0], delta=1.0)
    >>> [round(float(v), 6) for v in r["x"]]
    [0.8, 0.8]

    More penalty shrinks the solution, monotonically.

    >>> n1 = np.linalg.norm(boyd_regularized_ls(A, [2.0, 2.0], 0.1)["x"])
    >>> n2 = np.linalg.norm(boyd_regularized_ls(A, [2.0, 2.0], 10.0)["x"])
    >>> bool(n1 > n2)
    True

    Effective degrees of freedom fall from the rank toward zero as delta
    grows -- the sense in which ridge "uses fewer parameters".

    >>> bool(boyd_regularized_ls(A, [2.0, 2.0], 0.01)["effective_df"]
    ...      > boyd_regularized_ls(A, [2.0, 2.0], 100.0)["effective_df"])
    True

    >>> boyd_regularized_ls(A, [2.0, 2.0], -1.0)
    Traceback (most recent call last):
        ...
    ValueError: delta must be non-negative
    """
    Am = np.atleast_2d(np.asarray(A, dtype=float))
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    d = float(delta)
    if d < 0:
        raise ValueError("delta must be non-negative")
    if Am.shape[0] != bv.size:
        raise ValueError(f"A has {Am.shape[0]} rows but b has {bv.size}")
    n = Am.shape[1]
    stacked = np.vstack([Am, np.sqrt(d) * np.eye(n)])
    rhs = np.r_[bv, np.zeros(n)]
    x = np.linalg.lstsq(stacked, rhs, rcond=None)[0]
    resid = bv - Am @ x
    rss = float(resid @ resid)
    pen = float(d * (x @ x))
    sv = np.linalg.svd(Am, compute_uv=False)
    edf = float(np.sum(sv ** 2 / (sv ** 2 + d))) if sv.size else 0.0
    return RichResult(
        title="Regularized least squares",
        summary_lines=[("delta", d), ("rss", rss), ("penalty", pen),
                       ("effective df", edf)],
        payload={
            "x": x, "rss": rss, "penalty": pen, "objective": rss + pen,
            "effective_df": edf, "shrinkage": float(np.linalg.norm(x)),
            "residual": resid, "delta": d,
            "method": "boyd_regularized_ls",
        },
    )


def cheatsheet():
    return "cvxrgl: delta buys UNIQUENESS on a singular design, not just bias-variance"
