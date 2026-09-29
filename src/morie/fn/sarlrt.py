# morie.fn -- function file (rootcoder007/morie)
"""SAR likelihood-ratio test vs OLS."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sarlrt(ll_sar, ll_ols, df=1):
    r"""SAR likelihood-ratio test vs OLS.

    ``LR = 2 (ll_sar - ll_ols)`` referred to chi-square with ``df`` degrees of
    freedom (SAR against OLS: ``rho = 0``, 1 df); :func:`morie.fn.spdurbin.spatial_lr_test`.

    Parameters
    ----------
    ll_sar, ll_ols : float
        Maximised log-likelihoods of the unrestricted and restricted model.
    df : int
        Number of restrictions.

    Returns
    -------
    SpatialResult
        ``statistic`` (LR) and ``p_value`` (upper tail).

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Examples
    --------
    >>> r = sarlrt(-10.0, -12.5)
    >>> r.statistic, round(r.p_value, 12)
    (5.0, 0.025347318677)
    """
    lr = 2.0 * (float(ll_sar) - float(ll_ols))
    return SpatialResult(name="sarlrt", statistic=lr, p_value=sd._upper_p(lr, int(df)), extra={"df": int(df)})


sarlrt_fn = sarlrt


def cheatsheet() -> str:
    return "sarlrt(ll_sar, ll_ols, df) -> LR = 2(l1 - l0), chi-square p"
