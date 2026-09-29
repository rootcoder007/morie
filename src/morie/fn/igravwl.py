# morie.fn -- function file (rootcoder007/morie)
"""Gravity Wilson entropy model."""

from __future__ import annotations

import math

from . import _gravity as gv
from ._containers import SpatialResult


def igravwl(flows, mass_o, mass_d, dist, beta=1.0):
    r"""Gravity Wilson entropy model.

    Wilson's (1967) doubly constrained entropy-maximising model ``T_ij = A_i
    O_i B_j D_j exp(-beta c_ij)`` with balancing factors ``A_i = 1 / sum_j
    B_j D_j exp(-beta c_ij)`` and ``B_j = 1 / sum_i A_i O_i exp(-beta
    c_ij)`` iterated to convergence; ``O`` and ``D`` are the origin and
    destination totals and ``c`` the travel costs. The observed matrix, when
    given, is used for the fit: SRMSE (Knudsen and Fotheringham 1986) and
    the observed mean cost.

    Parameters
    ----------
    flows : array-like, shape (I, J) or None
        Observed flows (only for the fit statistics).
    mass_o : array-like, shape (I,)
        Origin totals ``O_i``.
    mass_d : array-like, shape (J,)
        Destination totals ``D_j`` (same sum).
    dist : array-like, shape (I, J)
        Travel costs ``c_ij``.
    beta : float
        Cost-decay parameter.

    Returns
    -------
    SpatialResult
        ``statistic`` is the modelled mean cost ``sum T c / sum T``;
        ``extra`` has ``T``, ``A``, ``B``, ``iterations`` and, with flows,
        ``srmse`` and ``observed_mean_cost``.

    References
    ----------
    Wilson, A. G. (1967). A statistical theory of spatial distribution models. *Transportation
    Research*, 1(3), 253-269.

    Wilson, A. G. (1970). *Entropy in Urban and Regional Modelling*. Pion, London.

    Knudsen, D. C. and Fotheringham, A. S. (1986). Matrix comparison, goodness-of-fit, and spatial
    interaction modeling. *International Regional Science Review*, 10(2), 127-147.

    Examples
    --------
    >>> r = igravwl(None, [10.0, 20.0], [15.0, 15.0], [[1.0, 2.0], [2.0, 1.0]], beta=0.5)
    >>> [[round(v, 10) for v in row] for row in r.extra["T"]]
    [[6.6213803873, 3.3786196127], [8.3786196127, 11.6213803873]]
    """
    og, dt, C = gv.vec(mass_o), gv.vec(mass_d), gv.mat(dist)
    T, A, B, it = gv.wilson(og, dt, C, float(beta))
    ni, nj = len(og), len(dt)
    tot = gv.ssum(v for r in T for v in r)
    extra = {"T": T, "A": A, "B": B, "iterations": it}
    if flows is not None:
        F = gv.mat(flows)
        rmse = math.sqrt(gv.ssum((F[i][j] - T[i][j]) ** 2 for i in range(ni) for j in range(nj)) / (ni * nj))
        extra["srmse"] = rmse / (gv.ssum(v for r in F for v in r) / (ni * nj))
        extra["observed_mean_cost"] = gv.ssum(F[i][j] * C[i][j] for i in range(ni) for j in range(nj)) / gv.ssum(
            v for r in F for v in r
        )
    mc = gv.ssum(T[i][j] * C[i][j] for i in range(ni) for j in range(nj)) / tot
    return SpatialResult(name="igravwl", statistic=mc, extra=extra)


igravwl_fn = igravwl


def cheatsheet() -> str:
    return "igravwl(flows, O, D, cost, beta) -> Wilson doubly constrained model"
