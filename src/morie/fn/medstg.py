# morie.fn -- function file (rootcoder007/morie)
"""Sequential / chain mediation X -> M1 -> M2 -> Y."""

import numpy as np

from ._richresult import RichResult

__all__ = ["sequential_mediation"]


def _ols(D, t):
    b, *_ = np.linalg.lstsq(D, t, rcond=None)
    return b


def sequential_mediation(x, m1, m2, y, c=None):
    r"""Serial (two-step chain) mediation effects.

    Fits the three-equation chain

    .. math::
        M_1 &= a_1 X, \\
        M_2 &= a_2 X + d\, M_1, \\
        Y   &= c' X + b_1 M_1 + b_2 M_2,

    and decomposes the total effect into the four paths

    - direct :math:`c'`,
    - through M1 only, :math:`a_1 b_1`,
    - through M2 only, :math:`a_2 b_2`,
    - serial :math:`a_1 d\, b_2` (the X -> M1 -> M2 -> Y chain).

    Parameters
    ----------
    x, m1, m2, y : array-like, shape (n,)
        Treatment, first mediator, second mediator, outcome.
    c : array-like, optional
        Baseline covariates entering every equation.

    Returns
    -------
    RichResult
        keys: ``direct``, ``via_m1``, ``via_m2``, ``serial``,
        ``indirect_total``, ``total``, ``paths`` dict with the raw
        coefficients ``a1 a2 d b1 b2 cprime``, ``n``, ``method``.

    References
    ----------
    Hayes, A. F. (2022). *Introduction to Mediation, Moderation, and
    Conditional Process Analysis* (3rd ed.). Guilford Press. Ch. 5
    (serial multiple mediator models).
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
        raise ValueError("too few observations for the chain regressions.")

    one = np.ones(n)
    a1 = float(_ols(np.column_stack([one, x, C]), m1)[1])
    b_m2 = _ols(np.column_stack([one, x, m1, C]), m2)
    a2, d = float(b_m2[1]), float(b_m2[2])
    b_y = _ols(np.column_stack([one, x, m1, m2, C]), y)
    cprime, b1, b2 = float(b_y[1]), float(b_y[2]), float(b_y[3])

    via_m1 = a1 * b1
    via_m2 = a2 * b2
    serial = a1 * d * b2
    ind = via_m1 + via_m2 + serial

    return RichResult(
        payload={
            "direct": cprime,
            "via_m1": via_m1,
            "via_m2": via_m2,
            "serial": serial,
            "indirect_total": ind,
            "total": cprime + ind,
            "paths": {"a1": a1, "a2": a2, "d": d, "b1": b1, "b2": b2, "cprime": cprime},
            "n": int(n),
            "method": "Serial mediation X -> M1 -> M2 -> Y (four-path decomposition)",
        }
    )


def cheatsheet():
    return "medstg: direct c', a1*b1, a2*b2, serial a1*d*b2"
