# morie.fn -- function file (rootcoder007/morie)
"""SEM likelihood-ratio test vs OLS."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def semlrt(ll_sem, ll_ols, df=1):
    r"""SEM likelihood-ratio test vs OLS.

    ``LR = 2 (ll_sem - ll_ols)`` referred to chi-square with ``df`` degrees of
    freedom (SEM against OLS: ``lambda = 0``, 1 df); :func:`morie.fn.spdurbin.spatial_lr_test`.

    Parameters
    ----------
    ll_sem, ll_ols : float
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
    >>> r = semlrt(-10.0, -12.5)
    >>> r.statistic, round(r.p_value, 12)
    (5.0, 0.025347318677)
    """
    lr = 2.0 * (float(ll_sem) - float(ll_ols))
    return SpatialResult(name="semlrt", statistic=lr, p_value=sd._upper_p(lr, int(df)), extra={"df": int(df)})


semlrt_fn = semlrt


def cheatsheet() -> str:
    return "semlrt(ll_sem, ll_ols, df) -> LR = 2(l1 - l0), chi-square p"
