# morie.fn -- function file (rootcoder007/morie)
"""Multi-mediator causal mediation analysis."""

import numpy as np

from ._richresult import RichResult

__all__ = ["multi_mediator_causal"]


def multi_mediator_causal(x, M, y, c=None):
    r"""Parallel mediation with an arbitrary number of mediators.

    Generalises the two-mediator model to k parallel mediators:
    each :math:`M_j` is regressed on X (and covariates) giving
    :math:`a_j`; Y is regressed on X and *all* mediators giving
    :math:`c'` and :math:`b_j`; the specific indirect effect of
    mediator j is :math:`a_j b_j` and the total indirect effect is
    :math:`\sum_j a_j b_j`.

    Parameters
    ----------
    x : array-like, shape (n,)
        Treatment.
    M : array-like, shape (n, k)
        Mediators, one column each.
    y : array-like, shape (n,)
        Outcome.
    c : array-like, optional
        Baseline covariates.

    Returns
    -------
    RichResult
        keys: ``indirect`` (k,), ``indirect_total``, ``direct``,
        ``total``, ``a`` (k,), ``b`` (k,), ``k``, ``n``, ``method``.

    References
    ----------
    Hayes, A. F. (2022). *Introduction to Mediation, Moderation, and
    Conditional Process Analysis* (3rd ed.). Guilford Press. Ch. 5.

    VanderWeele, T. J. (2015). *Explanation in Causal Inference*.
    Oxford University Press. Ch. 5 (multiple mediators; the joint
    versus specific indirect effects distinction).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    n, k = M.shape
    if x.size != n or y.size != n:
        raise ValueError("x, M, y must share their first dimension.")
    if c is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"c has {C.shape[0]} rows but x has {n}.")
    if n < k + C.shape[1] + 4:
        raise ValueError("too few observations for the mediator and outcome regressions.")

    def ols(D, t):
        b, *_ = np.linalg.lstsq(D, t, rcond=None)
        return b

    one = np.ones(n)
    Dx = np.column_stack([one, x, C])
    a = np.array([float(ols(Dx, M[:, j])[1]) for j in range(k)])
    by = ols(np.column_stack([one, x, M, C]), y)
    cprime = float(by[1])
    b = by[2 : 2 + k].astype(float)

    ind = a * b
    return RichResult(
        payload={
            "indirect": ind,
            "indirect_total": float(ind.sum()),
            "direct": cprime,
            "total": float(cprime + ind.sum()),
            "a": a,
            "b": b,
            "k": int(k),
            "n": int(n),
            "method": "Parallel multi-mediator model (specific + total indirect effects)",
        }
    )


def cheatsheet():
    return "mcausm: k parallel mediators -- indirect_j = a_j b_j, total = sum"
