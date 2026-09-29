# morie.fn -- function file (rootcoder007/morie)
"""SAR pseudo-R-squared (Nagelkerke)."""

from __future__ import annotations

import math

from ._containers import SpatialResult


def sarr2(ll_model, ll_null, n):
    r"""SAR pseudo-R-squared (Nagelkerke).

    Nagelkerke's (1991) rescaled likelihood-ratio R-squared of the spatial lag model
    against the intercept-only (or OLS) model: ``R2_CS = 1 - exp(2 (l_0 -
    l_1) / n)`` (Cox and Snell) divided by its maximum ``1 - exp(2 l_0 /
    n)``. Returns 0.0 when that maximum vanishes.

    Parameters
    ----------
    ll_model, ll_null : float
        Log-likelihoods of the fitted and the null model.
    n : int
        Number of observations.

    Returns
    -------
    SpatialResult
        ``statistic`` is Nagelkerke's R2; ``extra["cox_snell"]``.

    References
    ----------
    Nagelkerke, N. J. D. (1991). A note on a general definition of the coefficient of determination.
    *Biometrika*, 78(3), 691-692.

    Examples
    --------
    >>> round(sarr2(-20.0, -30.0, 50).statistic, 12)
    0.471776221068
    """
    n = float(n)
    cs = 1.0 - math.exp(2.0 / n * (float(ll_null) - float(ll_model)))
    mx = 1.0 - math.exp(2.0 / n * float(ll_null))
    return SpatialResult(name="sarr2", statistic=cs / mx if abs(mx) > 1e-12 else 0.0, extra={"cox_snell": cs})


sarr2_fn = sarr2


def cheatsheet() -> str:
    return "sarr2(ll_model, ll_null, n) -> Nagelkerke R2"
