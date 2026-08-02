# morie.fn -- function file (rootcoder007/morie)
"""Two-dimensional (parallel) mediation with M1 and M2."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["two_dimensional_mediation"]


def two_dimensional_mediation(x, m1, m2, y, c=None):
    r"""Parallel multiple-mediator model.

    Unlike the serial model, M1 and M2 are entered as *parallel*
    mediators -- neither is regressed on the other:

    .. math:: M_j = a_j X, \qquad
              Y = c' X + b_1 M_1 + b_2 M_2.

    Specific indirect effects are :math:`a_j b_j` and the total
    indirect effect is their sum. Comparing the two is the
    "which mediator carries more of the effect" question; the
    difference :math:`a_1 b_1 - a_2 b_2` is reported for that contrast.

    Parameters
    ----------
    x, m1, m2, y : array-like, shape (n,)
        Treatment, the two mediators, outcome.
    c : array-like, optional
        Baseline covariates.

    Returns
    -------
    RichResult
        keys: ``indirect_m1``, ``indirect_m2``, ``contrast``,
        ``indirect_total``, ``direct``, ``total``, ``paths``, ``n``,
        ``method``.

    References
    ----------
    Hayes, A. F. (2022). *Introduction to Mediation, Moderation, and
    Conditional Process Analysis* (3rd ed.). Guilford Press. Ch. 5
    (parallel multiple mediator models; specific indirect effects and
    their contrasts).
    """
    x = np.asarray(x, dtype=float).ravel()
    m1 = np.asarray(m1, dtype=float).ravel()
    m2 = np.asarray(m2, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if not (m1.size == n and m2.size == n and y.size == n):
        raise ValueError("x, m1, m2, y must have equal length.")
    if c is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"c has {C.shape[0]} rows but x has {n}.")
    if n < C.shape[1] + 6:
        raise ValueError("too few observations for the mediator regressions.")

    def ols(D, t):
        b, *_ = np.linalg.lstsq(D, t, rcond=None)
        return b

    one = np.ones(n)
    Dx = np.column_stack([one, x, C])
    a1 = float(ols(Dx, m1)[1])
    a2 = float(ols(Dx, m2)[1])
    by = ols(np.column_stack([one, x, m1, m2, C]), y)
    cprime, b1, b2 = float(by[1]), float(by[2]), float(by[3])

    i1, i2 = a1 * b1, a2 * b2
    return RichResult(
        payload={
            "indirect_m1": i1,
            "indirect_m2": i2,
            "contrast": i1 - i2,
            "indirect_total": i1 + i2,
            "direct": cprime,
            "total": cprime + i1 + i2,
            "paths": {"a1": a1, "a2": a2, "b1": b1, "b2": b2, "cprime": cprime},
            "n": int(n),
            "method": "Parallel two-mediator model (specific indirect effects + contrast)",
        }
    )


def cheatsheet():
    return "tdmed: parallel mediators -- a1*b1, a2*b2, their contrast, direct c'"
