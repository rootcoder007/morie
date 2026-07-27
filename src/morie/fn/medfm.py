# morie.fn -- function file (rootcoder007/morie)
"""Pearl's mediation formula (discrete mediator)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["mediation_formula"]


def mediation_formula(x, m, y, x1=None, x0=None):
    r"""Natural direct/indirect effects by the mediation formula.

    .. math::

        NDE &= \sum_m \{E[Y|x_1, m] - E[Y|x_0, m]\}\,P(m|x_0) \\
        NIE &= \sum_m E[Y|x_1, m]\,\{P(m|x_1) - P(m|x_0)\}

    with :math:`TE = NDE + NIE` holding by construction for this
    decomposition (Pearl 2001; VanderWeele 2015, Ch. 2). Everything is
    estimated by empirical cell means over a DISCRETE treatment and
    mediator; identification needs sequential ignorability, which data
    cannot certify.

    This replaces a placeholder that averaged its first argument.

    Parameters
    ----------
    x, m, y : array-like, shape (n,)
        Discrete treatment, discrete mediator, outcome.
    x1, x0 : scalars, optional
        Treatment levels to contrast; default the two most common.

    Returns
    -------
    RichResult
        keys: ``nde``, ``nie``, ``te``, ``x1``, ``x0``,
        ``incomplete_cells``, ``n``, ``method``.

    References
    ----------
    Pearl, J. (2001). Direct and indirect effects. *Proc. 17th
    Conference on Uncertainty in Artificial Intelligence*, 411-420.
    VanderWeele, T. J. (2015). *Explanation in Causal Inference*.
    Oxford UP. Ch. 2.
    """
    xa = np.asarray(x).ravel()
    ma = np.asarray(m).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = xa.size
    if not (ma.size == n and ya.size == n):
        raise ValueError(f"x, m and y must share a length; got {n}, {ma.size}, {ya.size}.")
    levels = list(np.unique(xa))
    if x1 is None or x0 is None:
        if len(levels) < 2:
            raise ValueError("x must take at least 2 values.")
        counts = [(np.sum(xa == v), v) for v in levels]
        counts.sort(reverse=True, key=lambda t: t[0])
        x1 = counts[0][1] if x0 is None and x1 is None else x1
        x0 = counts[1][1] if x0 is None else x0
        if x1 is None:
            x1 = counts[0][1]
    for v in (x1, x0):
        if not np.any(xa == v):
            raise ValueError(f"treatment level {v!r} does not occur in x.")

    m_levels = np.unique(ma)
    p_m_x1 = np.array([np.mean(ma[xa == x1] == mv) for mv in m_levels])
    p_m_x0 = np.array([np.mean(ma[xa == x0] == mv) for mv in m_levels])

    incomplete = []
    Ey = {}
    for xv in (x1, x0):
        for mv in m_levels:
            sel = (xa == xv) & (ma == mv)
            if sel.any():
                Ey[(xv, mv)] = float(ya[sel].mean())
            else:
                Ey[(xv, mv)] = np.nan
                incomplete.append((xv, mv))

    nde = float(np.nansum([(Ey[(x1, mv)] - Ey[(x0, mv)]) * p_m_x0[i] for i, mv in enumerate(m_levels)]))
    nie = float(np.nansum([Ey[(x1, mv)] * (p_m_x1[i] - p_m_x0[i]) for i, mv in enumerate(m_levels)]))
    return RichResult(
        payload={
            "nde": nde,
            "nie": nie,
            "te": nde + nie,
            "x1": x1,
            "x0": x0,
            "incomplete_cells": incomplete,
            "n": int(n),
            "method": "Pearl mediation formula (discrete, empirical cell means)",
        }
    )


def cheatsheet():
    return "medfm: Pearl mediation formula, NDE/NIE for a discrete mediator"
