# morie.fn -- function file (rootcoder007/morie)
"""Minimax -- Boyd & Vandenberghe Sec. 4.3 / 6.1."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_minimax"]


def boyd_minimax(A, b=None):
    r"""Minimise :math:`\max_i (a_i^\top x + b_i)`, the pointwise maximum
    of affine functions.

    A pointwise maximum of convex functions is convex, so this is a convex
    problem however many pieces there are -- and it is an LP, via the
    epigraph trick :math:`\min t` s.t. :math:`a_i^\top x + b_i \le t`.
    A pointwise MINIMUM of convex functions is not convex, which is why
    maximin and minimax are not symmetric in difficulty.

    At the optimum several pieces TIE at the maximum. That is not a
    coincidence: if a single piece were strictly largest, x could move to
    reduce it, so the solution is characterised by the tie. The count of
    active pieces is reported for exactly that reason.

    Parameters
    ----------
    A : array-like
        Rows are the affine coefficient vectors.
    b : array-like, optional
        Offsets; zero when omitted.

    Returns
    -------
    RichResult
        ``x``, ``value``, ``active`` (pieces attaining the max),
        ``n_active``, ``ties``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Minimising the larger of two opposed affine functions balances them.

    >>> import numpy as np
    >>> A = np.array([[1.0], [-1.0]])
    >>> r = boyd_minimax(A, [0.0, 2.0])
    >>> round(float(r["x"][0]), 6), round(r["value"], 6)
    (1.0, 1.0)

    Both pieces tie at the optimum, which is what characterises it.

    >>> int(r["n_active"]), bool(r["ties"])
    (2, True)

    With three pieces in two variables the solution still ties, on as
    many pieces as it takes to pin x down.

    >>> A3 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    >>> s = boyd_minimax(A3, [0.0, 0.0, 1.0])
    >>> bool(s["n_active"] >= 2)
    True
    """
    from ._sci_core import linprog

    Am = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = Am.shape
    bv = (np.zeros(m) if b is None
          else np.atleast_1d(np.asarray(b, dtype=float)).ravel())
    if bv.size != m:
        raise ValueError(f"A has {m} rows but b has {bv.size}")
    # Epigraph: min t s.t. a_i'x + b_i <= t.
    c = np.r_[np.zeros(n), 1.0]
    A_ub = np.hstack([Am, -np.ones((m, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=-bv,
                  bounds=[(None, None)] * (n + 1), method="highs")
    if res.status != 0:
        return RichResult(
            title="Minimax",
            summary_lines=[("status", str(res.message))],
            warnings=["the minimax LP did not solve; the pointwise maximum "
                      "may be unbounded below"],
            payload={"x": np.full(n, np.nan), "value": float("nan"),
                     "active": np.zeros(m, dtype=bool), "n_active": 0,
                     "ties": False, "method": "boyd_minimax"})
    x = np.asarray(res.x[:n], dtype=float)
    vals = Am @ x + bv
    t = float(vals.max())
    active = np.abs(vals - t) <= 1e-08 * max(1.0, abs(t))
    return RichResult(
        title="Minimax",
        summary_lines=[("n", int(n)), ("pieces", int(m)), ("value", t),
                       ("active pieces", int(active.sum()))],
        payload={
            "x": x, "value": t, "active": active,
            "n_active": int(active.sum()),
            "ties": bool(active.sum() > 1), "piece_values": vals,
            "method": "boyd_minimax",
        },
    )


def cheatsheet():
    return "cvxmnr: max of affine is convex (an LP); MIN of convex is not -- minimax and maximin differ"


# compact alias per ledger/NAMING.md
boydminimax = boyd_minimax
