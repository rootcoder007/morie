# morie.fn -- function file (rootcoder007/morie)
"""Spatial weights coding schemes (spdep styles), Sinkhorn scaling and weights summaries."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["weights_style", "sinkhorn_weights", "weights_summary"]


def _mat(W):
    M = [[float(v) for v in row] for row in W]
    n = len(M)
    if any(len(r) != n for r in M):
        raise ValueError("W must be square")
    return M


def weights_style(W, style: str = "W"):
    r"""Recode a spatial weights matrix with an ``spdep::nb2listw`` style.

    With ``g`` the given weights, ``n*`` the number of units with at least
    one neighbour (spdep's ``zero.policy`` count) and ``D = sum g``: ``W`` row-standardised; ``B`` binary
    ``1(g_ij != 0)``; ``C`` globally standardised ``n* g / D``; ``U`` ``g /
    D``; ``minmax`` ``g / min(max row sum, max column sum)`` (Kelejian and
    Prucha 2010); ``S`` variance-stabilising ``n (g_ij / q_i) / Q`` with
    ``q_i = sqrt(sum_j g_ij^2)`` and ``Q`` the sum of the ``g_ij / q_i``
    (Tiefelsdorf, Griffith and Boots 1999).  Rows without neighbours stay 0.

    References
    ----------
    Tiefelsdorf, M., Griffith, D. A. and Boots, B. (1999). A
    variance-stabilizing coding scheme for spatial link matrices.
    *Environment and Planning A*, 31(1), 165-180.
    Kelejian, H. H. and Prucha, I. R. (2010). Specification and estimation
    of spatial autoregressive models with autoregressive and
    heteroskedastic disturbances. *Journal of Econometrics*, 157(1), 53-67.

    Examples
    --------
    >>> weights_style([[0, 1, 1], [1, 0, 0], [1, 0, 0]], "W")
    [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    """
    M = _mat(W)
    n = len(M)
    card = [sum(1 for v in r if v != 0) for r in M]
    eff = sum(1 for c in card if c > 0)
    if style == "B":
        return [[1.0 if v != 0 else 0.0 for v in r] for r in M]
    if style == "W":
        return [[v / ssum(r) if card[i] else 0.0 for v in r] for i, r in enumerate(M)]
    D = ssum(v for r in M for v in r)
    if style in ("C", "U", "minmax", "S") and not D > 0:
        raise ValueError("the weights must have a positive sum")
    if style == "C":
        return [[eff * v / D for v in r] for r in M]
    if style == "U":
        return [[v / D for v in r] for r in M]
    if style == "minmax":
        mm = min(max(ssum(r) for r in M), max(ssum(M[i][j] for i in range(n)) for j in range(n)))
        return [[v / mm for v in r] for r in M]
    if style == "S":
        q = [math.sqrt(ssum(v * v for v in r)) for r in M]
        G = [[v / q[i] if q[i] > 0 else 0.0 for v in r] for i, r in enumerate(M)]
        Q = ssum(v for r in G for v in r)
        return [[n * v / Q for v in r] for r in G]
    raise ValueError("style must be W, B, C, U, minmax or S")


def sinkhorn_weights(W, *, tol: float = 1e-12, maxit: int = 10000) -> RichResult:
    r"""Doubly-stochastic scaling ``D_1 W D_2`` by alternate row and column normalisation (Sinkhorn 1964).

    Converges for a non-negative matrix with total support (every nonzero
    entry on a positive diagonal, e.g. a contiguity matrix whose graph has a
    cycle cover); returns the scaled matrix, the
    number of sweeps and the final maximum deviation of row and column sums
    from 1.

    References
    ----------
    Sinkhorn, R. (1964). A relationship between arbitrary positive matrices
    and doubly stochastic matrices. *The Annals of Mathematical Statistics*,
    35(2), 876-879.

    Examples
    --------
    >>> r = sinkhorn_weights([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> [round(v, 6) for v in r.W[0]]
    [0.0, 0.5, 0.5]
    """
    M = _mat(W)
    n = len(M)
    if any(v < 0 for r in M for v in r):
        raise ValueError("W must be non-negative")
    dev = float("inf")
    sweeps = 0
    for _ in range(maxit):
        sweeps += 1
        M = [[v / ssum(r) if ssum(r) > 0 else 0.0 for v in r] for r in M]
        cs = [ssum(M[i][j] for i in range(n)) for j in range(n)]
        M = [[M[i][j] / cs[j] if cs[j] > 0 else 0.0 for j in range(n)] for i in range(n)]
        dev = max(abs(ssum(r) - 1.0) for r in M)
        if dev < tol:
            break
    return RichResult(payload={"W": M, "iterations": sweeps, "deviation": dev})


def weights_summary(W, *, eigen: bool = True) -> RichResult:
    r"""Summary of a spatial weights matrix.

    ``n``, ``nonzero`` links, ``pct_nonzero`` (density, %), ``avg_links``,
    ``cardinality`` (neighbours per unit, ``spdep::card``), ``islands``
    (units without neighbours), ``symmetric``, ``row_stochastic``, the
    Cliff-Ord constants ``S0 = sum w_ij``, ``S1 = sum (w_ij + w_ji)^2 / 2``,
    ``S2 = sum_i (w_i. + w_.i)^2`` (``spdep::spweights.constants``),
    ``trace_W2 = tr(W^2)``, ``trace_WtW = tr(W'W)``, ``frobenius``,
    ``diag_dominant`` (``|w_ii| >= sum_{j != i} |w_ij|`` for all ``i``),
    ``asymmetry`` (Frobenius norm of ``(W - W')/2`` over that of ``W``),
    ``lower_triangle`` and, with ``eigen``, the ``eigenvalues`` of ``W``
    sorted by real part (``spatialreg::eigenw``; complex for asymmetric
    ``W``) (Cliff and Ord 1981).

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> s = weights_summary([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> s.nonzero, s.cardinality, s.S0, s.S1, s.S2
    (4, [1, 2, 1], 4.0, 8.0, 24.0)
    """
    M = _mat(W)
    n = len(M)
    card = [sum(1 for j, v in enumerate(r) if v != 0 and j != i) for i, r in enumerate(M)]
    nz = sum(1 for r in M for v in r if v != 0)
    rs = [ssum(r) for r in M]
    cs = [ssum(M[i][j] for i in range(n)) for j in range(n)]
    fro = math.sqrt(ssum(v * v for r in M for v in r))
    asym = math.sqrt(ssum(((M[i][j] - M[j][i]) / 2.0) ** 2 for i in range(n) for j in range(n)))
    out = {
        "n": n,
        "nonzero": nz,
        "pct_nonzero": 100.0 * nz / (n * n),
        "avg_links": nz / n,
        "cardinality": card,
        "islands": [i for i in range(n) if card[i] == 0],
        "symmetric": all(M[i][j] == M[j][i] for i in range(n) for j in range(i)),
        "row_stochastic": all(abs(r - 1.0) < 1e-12 for r in rs) and all(v >= 0 for r in M for v in r),
        "S0": ssum(rs),
        "S1": 0.5 * ssum((M[i][j] + M[j][i]) ** 2 for i in range(n) for j in range(n)),
        "S2": ssum((rs[i] + cs[i]) ** 2 for i in range(n)),
        "trace_W2": ssum(M[i][j] * M[j][i] for i in range(n) for j in range(n)),
        "trace_WtW": ssum(v * v for r in M for v in r),
        "frobenius": fro,
        "diag_dominant": all(abs(M[i][i]) >= ssum(abs(M[i][j]) for j in range(n) if j != i) for i in range(n)),
        "asymmetry": asym / fro if fro > 0 else 0.0,
        "lower_triangle": [[M[i][j] if j < i else 0.0 for j in range(n)] for i in range(n)],
    }
    if eigen:
        from ._array_core import linalg

        ev = [complex(v) for v in linalg.eigvals([r[:] for r in M])]
        ev = sorted(ev, key=lambda z: (z.real, z.imag))
        out["eigenvalues"] = [z.real for z in ev] if all(abs(z.imag) < 1e-12 for z in ev) else ev
    return RichResult(payload=out)


def cheatsheet() -> str:
    return "weights_style / sinkhorn_weights / weights_summary -> spdep weight styles, Sinkhorn scaling, summaries."
