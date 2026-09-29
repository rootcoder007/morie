# morie.fn -- function file (rootcoder007/morie)
"""SAC likelihood-ratio test vs SAR/SEM."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def saclrt(ll_sac, ll_sar, df=1):
    r"""SAC likelihood-ratio test vs SAR/SEM.

    ``LR = 2 (ll_sac - ll_sar)`` referred to chi-square with ``df`` degrees of
    freedom (SAC against the SAR or SEM it nests: one autoregressive parameter, 1 df); :func:`morie.fn.spdurbin.spatial_lr_test`.

    Parameters
    ----------
    ll_sac, ll_sar : float
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
    >>> r = saclrt(-10.0, -12.5)
    >>> r.statistic, round(r.p_value, 12)
    (5.0, 0.025347318677)
    """
    lr = 2.0 * (float(ll_sac) - float(ll_sar))
    return SpatialResult(name="saclrt", statistic=lr, p_value=sd._upper_p(lr, int(df)), extra={"df": int(df)})


saclrt_fn = saclrt


def cheatsheet() -> str:
    return "saclrt(ll_sac, ll_sar, df) -> LR = 2(l1 - l0), chi-square p"
