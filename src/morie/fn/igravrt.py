# morie.fn -- function file (rootcoder007/morie)
"""Gravity retail potential (Huff model)."""

from __future__ import annotations

from . import _gravity as gv
from ._containers import SpatialResult


def igravrt(mass, dist, beta=2.0, demand=None):
    r"""Gravity retail potential (Huff model).

    Huff's (1963) probability that a consumer at ``i`` patronises centre
    ``j``: ``P_ij = (M_j / d_ij^beta) / sum_k (M_k / d_ik^beta)`` with
    attractiveness ``M`` (e.g. floor space) and distance-decay exponent
    ``beta``; expected patronage of centre ``j`` is ``sum_i demand_i
    P_ij`` (unit demand by default).

    Parameters
    ----------
    mass : array-like, shape (J,)
        Attractiveness of the J centres.
    dist : array-like, shape (I, J)
        Distances from the I consumer locations to the centres.
    beta : float
        Distance-decay exponent.
    demand : array-like, shape (I,), optional
        Spending or population at each consumer location.

    Returns
    -------
    SpatialResult
        ``statistic`` is the expected patronage of the first centre;
        ``extra`` has ``probabilities`` (I x J) and ``patronage`` (J).

    References
    ----------
    Huff, D. L. (1963). A probabilistic analysis of shopping center trade areas. *Land Economics*,
    39(1), 81-90.

    Examples
    --------
    >>> r = igravrt([10.0, 20.0], [[1.0, 2.0], [3.0, 1.0]])
    >>> [[round(v, 12) for v in row] for row in r.extra["probabilities"]]
    [[0.666666666667, 0.333333333333], [0.052631578947, 0.947368421053]]
    """
    P = gv.huff(mass, dist, float(beta))
    w = [1.0] * len(P) if demand is None else gv.vec(demand)
    pat = [gv.ssum(w[i] * P[i][j] for i in range(len(P))) for j in range(len(P[0]))]
    return SpatialResult(name="igravrt", statistic=pat[0], extra={"probabilities": P, "patronage": pat})


igravrt_fn = igravrt


def cheatsheet() -> str:
    return "igravrt(mass, dist, beta, demand) -> Huff patronage probabilities"
