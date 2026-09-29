# morie.fn -- function file (rootcoder007/morie)
"""LISA (local indicators spatial association)."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def laclisa(y, W):
    r"""LISA (local indicators spatial association).

    Local Moran's ``I_i = (z_i / m2) sum_j w_ij z_j``, ``z = y - ybar``,
    ``m2 = sum z^2 / n`` (Anselin 1995), with the conditional randomisation
    moments of Sokal, Oden and Thomson (1998): ``E(I_i) = -z_i^2 W_i / ((n -
    1) m2)`` and ``Var(I_i) = (z_i / m2)^2 n / (n - 2) (W_i2 - W_i^2 / (n -
    1)) (m2 - z_i^2 / (n - 1))``, ``W_i = sum_j w_ij``, ``W_i2 = sum_j
    w_ij^2``; two-sided normal p-values. Exactly ``spdep::localmoran`` with
    its defaults (``conditional = TRUE``, ``mlvar = TRUE``).

    Parameters
    ----------
    y : array-like, shape (n,)
        Variable observed on the n lattice units.
    W : array-like, shape (n, n)
        Spatial weights (zero diagonal).

    Returns
    -------
    SpatialResult
        ``statistic`` is the sum of the ``I_i`` divided by ``S0`` (global
        Moran's I when ``S0 = n``); ``local_values`` the ``I_i``; ``extra``
        has ``expected``, ``variance``, ``z``, ``p_value`` lists.

    References
    ----------
    Anselin, L. (1995). Local indicators of spatial association -- LISA. *Geographical Analysis*,
    27(2), 93-115.

    Sokal, R. R., Oden, N. L. and Thomson, B. A. (1998). Local spatial autocorrelation in a
    biological model. *Geographical Analysis*, 30(4), 331-354.

    Bivand, R. S. and Wong, D. W. S. (2018). Comparing implementations of global and local
    indicators of spatial association. *TEST*, 27(3), 716-748.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = laclisa([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W)
    >>> [round(v, 10) for v in r.local_values[:3]]
    [-0.8452722063, -0.7163323782, -1.0808022923]
    """
    Wm = lat.mat(W)
    r = lat.local_moran(lat.vec(y), Wm)
    S0 = lat.ssum(v for row in Wm for v in row)
    return SpatialResult(
        name="laclisa",
        statistic=lat.ssum(r["Ii"]) / S0,
        local_values=r["Ii"],
        extra={"expected": r["expected"], "variance": r["variance"], "z": r["z"], "p_value": r["p_value"]},
    )


laclisa_fn = laclisa


def cheatsheet() -> str:
    return "laclisa(y, W) -> local Moran's I with conditional moments (spdep::localmoran)"
