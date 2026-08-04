# morie.fn -- function file (rootcoder007/morie)
"""Pickands estimator of the extreme-value index."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ev_pickands", "evt_pickands_estimator"]


def ev_pickands(x, k=None):
    r"""The Pickands (1975) estimator,

    .. math:: \hat\xi_P = \frac1{\log 2}\log
              \frac{X_{(n-k+1)} - X_{(n-2k+1)}}
                   {X_{(n-2k+1)} - X_{(n-4k+1)}},

    built from three upper order statistics at spacings k, 2k, 4k.

    Its virtue over Hill is GENERALITY: it is consistent for EVERY
    real :math:`\xi` -- heavy (Frechet), light (Gumbel, xi = 0) and
    bounded (Weibull, xi < 0) tails alike -- because it uses spacings
    rather than logarithms of levels. The price is efficiency: its
    asymptotic variance,

    .. math:: \frac{\xi^2(2^{2\xi+1}+1)}
              {(2(2^\xi-1)\log 2)^2}

    (de Haan and Ferreira 2006, Thm. 3.3.5; finite limit
    :math:`3/(4\log^2 2)` at :math:`\xi = 0`), is far above Hill's
    :math:`\xi^2` at the same k when both are valid, so Hill wins
    whenever its xi > 0 assumption holds and Pickands is the tool
    precisely when it might not.

    Parameters
    ----------
    x : array-like
        Sample.
    k : int, optional
        Spacing parameter; needs ``4k <= n``. Default
        ``n // 8``.

    Returns
    -------
    RichResult
        keys: ``xi``, ``se``, ``k``, ``order_stats_used``,
        ``valid_for``, ``versus_hill``, ``n``, ``method``.

    References
    ----------
    Pickands, J. (1975), "Statistical inference using extreme order
    statistics", *Annals of Statistics* 3:119-131. de Haan, L. and
    Ferreira, A. (2006), *Extreme Value Theory*, Springer,
    Thm. 3.3.5, for the variance.
    """
    xv = np.sort(np.asarray(x, dtype=float).ravel())
    n = xv.size
    if n < 8:
        raise ValueError(f"need at least 8 observations, got {n}.")
    kk = n // 8 if k is None else int(k)
    if not 1 <= 4 * kk <= n:
        raise ValueError(f"need 4k <= n; got k = {kk}, n = {n}.")
    a = xv[n - kk]
    b = xv[n - 2 * kk]
    c = xv[n - 4 * kk]
    if not (a > b > c):
        raise ValueError("the three order statistics are tied; the spacing "
                         "ratio is undefined.")
    xi = float(np.log((a - b) / (b - c)) / np.log(2))
    # asymptotic variance (de Haan-Ferreira Thm 3.3.5)
    if abs(xi) < 1e-8:
        avar = 3.0 / (4.0 * np.log(2) ** 2)
    else:
        avar = (xi ** 2 * (2.0 ** (2 * xi + 1) + 1)
                / (2 * (2.0 ** xi - 1) * np.log(2)) ** 2)
    return RichResult(payload={
        "xi": xi, "se": float(np.sqrt(avar / kk)),
        "k": kk,
        "order_stats_used": (float(a), float(b), float(c)),
        "valid_for": "every real xi -- heavy, light and bounded tails alike",
        "versus_hill": "far less efficient than Hill where Hill is valid "
                       "(xi > 0); the tool for when the sign of xi is "
                       "itself in question",
        "n": int(n),
        "method": "Pickands (1975): log spacing ratio at k, 2k, 4k over log 2"})


def cheatsheet():
    return "evpick: works for every xi, pays in variance -- Hill wins when xi > 0 is known"


#: Catalogue alias for :func:`ev_pickands`.
evt_pickands_estimator = ev_pickands


# compact alias per ledger/NAMING.md
evpickands = ev_pickands
