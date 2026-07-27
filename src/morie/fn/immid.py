# morie.fn -- function file (rootcoder007/morie)
"""Index of moderated mediation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["index_moderated_mediation"]


def index_moderated_mediation(x, m, y, w, c=None, w_values=None):
    r"""Hayes' index of moderated mediation (first-stage moderation).

    With the a-path moderated by W,

    .. math::
        M &= a_0 + a_1 X + a_2 W + a_3 XW, \\
        Y &= c' X + b M,

    the conditional indirect effect is :math:`(a_1 + a_3 W) b`, a
    linear function of W whose slope

    .. math:: \text{index of moderated mediation} = a_3 b

    is the quantity to test: the indirect effect depends on W if and
    only if the index is nonzero. Comparing conditional effects at two
    W levels without testing the index is the error the index was
    introduced to fix.

    Parameters
    ----------
    x, m, y, w : array-like, shape (n,)
        Treatment, mediator, outcome, moderator.
    c : array-like, optional
        Baseline covariates.
    w_values : array-like, optional
        Moderator levels at which to report the conditional indirect
        effect. Default: the 16th, 50th and 84th percentiles of W.

    Returns
    -------
    RichResult
        keys: ``index``, ``conditional_indirect`` (matching
        ``w_values``), ``w_values``, ``a1``, ``a3``, ``b``,
        ``direct``, ``n``, ``method``.

    References
    ----------
    Hayes, A. F. (2015). An index and test of linear moderated
    mediation. *Multivariate Behavioral Research*, 50(1), 1-22.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = x.size
    if not (m.size == n and y.size == n and w.size == n):
        raise ValueError("x, m, y, w must have equal length.")
    if c is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"c has {C.shape[0]} rows but x has {n}.")
    if n < C.shape[1] + 7:
        raise ValueError("too few observations for the moderated mediation regressions.")

    def ols(D, t):
        b, *_ = np.linalg.lstsq(D, t, rcond=None)
        return b

    one = np.ones(n)
    a = ols(np.column_stack([one, x, w, x * w, C]), m)
    a1, a3 = float(a[1]), float(a[3])
    by = ols(np.column_stack([one, x, m, C]), y)
    cprime, b = float(by[1]), float(by[2])

    wv = np.percentile(w, [16, 50, 84]) if w_values is None else np.asarray(w_values, dtype=float).ravel()
    cond = (a1 + a3 * wv) * b

    return RichResult(
        payload={
            "index": float(a3 * b),
            "conditional_indirect": cond,
            "w_values": wv,
            "a1": a1,
            "a3": a3,
            "b": b,
            "direct": cprime,
            "n": int(n),
            "method": "Index of moderated mediation a3*b (first-stage moderation)",
        }
    )


def cheatsheet():
    return "immid: index = a3*b; conditional indirect effect = (a1 + a3 W) b"
