# morie.fn -- function file (rootcoder007/morie)
"""LISA high-low outlier identification."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def laclihl(y, W, p_thr=0.05):
    r"""LISA high-low outlier identification.

    Units whose local Moran's ``I_i`` is significant (two-sided p-value
    below ``p_thr``, conditional randomisation moments as
    ``spdep::localmoran``) and that fall in the high-low quadrant of the
    Moran scatterplot: ``z_i > 0`` and ``(Wz)_i <= 0`` (Anselin 1995; quadrants on ``z = y - ybar``
    and its lag ``Wz``, the ``pysal`` quadrants of ``spdep``). No
    multiple-testing adjustment is applied; pass a smaller ``p_thr`` for
    one.

    Parameters
    ----------
    y : array-like, shape (n,)
        Variable observed on the n lattice units.
    W : array-like, shape (n, n)
        Spatial weights (zero diagonal).
    p_thr : float
        Significance threshold.

    Returns
    -------
    SpatialResult
        ``statistic`` is the number of such units; ``extra`` has their
        ``indices`` and the local ``p_value`` and ``Ii`` lists.

    References
    ----------
    Anselin, L. (1995). Local indicators of spatial association -- LISA. *Geographical Analysis*,
    27(2), 93-115.

    Sokal, R. R., Oden, N. L. and Thomson, B. A. (1998). Local spatial autocorrelation in a
    biological model. *Geographical Analysis*, 30(4), 331-354.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = laclihl([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W, p_thr=0.5)
    >>> r.statistic, r.extra["indices"]
    (1.0, [1])
    """
    yv, Wm = lat.vec(y), lat.mat(W)
    r = lat.local_moran(yv, Wm)
    q, _z, _lz = lat.quadrants(yv, Wm)
    idx = [i for i in range(len(yv)) if q[i] == 4 and r["p_value"][i] < p_thr]
    return SpatialResult(
        name="laclihl",
        statistic=float(len(idx)),
        local_values=r["Ii"],
        extra={"indices": idx, "p_value": r["p_value"], "Ii": r["Ii"], "quadrant": q},
    )


laclihl_fn = laclihl


def cheatsheet() -> str:
    return "laclihl(y, W, p_thr) -> significant high-low LISA units"
