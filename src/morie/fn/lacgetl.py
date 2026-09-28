# morie.fn -- function file (rootcoder007/morie)
"""Local Getis-Ord Gi and Gi* statistics as z-values (Getis and Ord 1992; Ord and Getis 1995)."""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as stats
from ._containers import SpatialResult


def lacgetl(y, W, star: bool = True) -> SpatialResult:
    r"""Local Getis-Ord statistic of each unit, standardised.

    ``Gi*`` (``star``, the unit counted in its own neighbourhood; put the
    self-weights on the diagonal of ``W``, e.g. ``1`` before any coding):

    .. math::

        z_i = \frac{\sum_j w_{ij} x_j - W_i \bar x}
                   {s \sqrt{(n S_{1i} - W_i^2)/(n - 1)}}

    with ``W_i = sum_j w_ij``, ``S_1i = sum_j w_ij^2``, ``xbar`` and the
    population variance ``s^2`` over all units.  ``Gi`` (``star=False``)
    excludes unit ``i``: the diagonal of ``W`` is ignored, ``xbar_i`` and
    ``s_i^2`` are over the other ``n - 1`` units and the variance factor is
    ``((n - 1) S_1i - W_i^2)/(n - 2)``.  Exactly ``spdep::localG`` (with
    ``include.self`` for ``Gi*``).  Large positive ``z`` marks a hot spot.

    :param y: Values (n,), normally non-negative.
    :param W: Spatial weights (n, n).
    :param star: ``Gi*`` (default) or ``Gi``.
    :return: :class:`SpatialResult`: ``local_values`` the z-values,
        ``statistic`` the largest absolute z, ``extra`` has two-sided
        ``p_values``.

    References
    ----------
    Getis, A. and Ord, J. K. (1992). The analysis of spatial association by
    use of distance statistics. *Geographical Analysis*, 24(3), 189-206.
    Ord, J. K. and Getis, A. (1995). Local spatial autocorrelation
    statistics: distributional issues and an application. *Geographical
    Analysis*, 27(4), 286-306.

    Examples
    --------
    >>> W = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]
    >>> [round(z, 6) for z in lacgetl([1.0, 2.0, 4.0, 8.0], W).local_values]
    [-1.453631, -1.585258, 1.025755, 1.453631]
    """
    x = [float(v) for v in np.asarray(y, dtype=float).tolist()]
    Wm = [[float(v) for v in r] for r in np.asarray(W, dtype=float).tolist()]
    n = len(x)
    if len(Wm) != n or any(len(r) != n for r in Wm):
        raise ValueError("W must be n x n with n = len(y)")
    if n < 3:
        raise ValueError("need at least three units")
    tot, tot2 = sum(x), sum(v * v for v in x)
    z = []
    for i in range(n):
        w = [Wm[i][j] if (star or j != i) else 0.0 for j in range(n)]
        Wi = sum(w)
        S1 = sum(v * v for v in w)
        lx = sum(w[j] * x[j] for j in range(n))
        if star:
            xb = tot / n
            s2 = sum((v - xb) ** 2 for v in x) / n
            VG = s2 * (n * S1 - Wi * Wi) / (n - 1)
        else:
            xb = (tot - x[i]) / (n - 1)
            s2 = (tot2 - x[i] ** 2) / (n - 1) - xb * xb
            VG = s2 * ((n - 1) * S1 - Wi * Wi) / (n - 2)
        z.append((lx - Wi * xb) / math.sqrt(VG) if VG > 0 else float("nan"))
    p = [float(2.0 * stats.norm.sf(abs(v))) if v == v else float("nan") for v in z]
    finite = [abs(v) for v in z if v == v]
    return SpatialResult(
        name="lacgetl",
        statistic=max(finite) if finite else float("nan"),
        p_value=None,
        local_values=z,
        extra={"p_values": p, "star": bool(star)},
    )


lacgetl_fn = lacgetl


def cheatsheet() -> str:
    return "lacgetl(y, W, star) -> local Getis-Ord Gi* / Gi z-values (spdep::localG)."
