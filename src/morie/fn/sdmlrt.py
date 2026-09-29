# morie.fn -- function file (rootcoder007/morie)
"""SDM likelihood-ratio test vs SAR."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sdmlrt(ll_sdm, ll_sar, df=2):
    r"""SDM likelihood-ratio test vs SAR.

    ``LR = 2 (ll_sdm - ll_sar)`` referred to chi-square with ``df`` degrees of
    freedom (SDM against SAR: ``theta = 0``, ``df`` = number of lagged covariates); :func:`morie.fn.spdurbin.spatial_lr_test`.

    Parameters
    ----------
    ll_sdm, ll_sar : float
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
    >>> r = sdmlrt(-10.0, -12.5)
    >>> r.statistic, round(r.p_value, 12)
    (5.0, 0.082084998624)
    """
    lr = 2.0 * (float(ll_sdm) - float(ll_sar))
    return SpatialResult(name="sdmlrt", statistic=lr, p_value=sd._upper_p(lr, int(df)), extra={"df": int(df)})


sdmlrt_fn = sdmlrt


def cheatsheet() -> str:
    return "sdmlrt(ll_sdm, ll_sar, df) -> LR = 2(l1 - l0), chi-square p"
