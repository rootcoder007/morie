# morie.fn -- function file (rootcoder007/morie)
"""Spatial NB likelihood-ratio test."""

from __future__ import annotations

import math

from ._containers import SpatialResult


def scnblrt(ll_nb, ll_pois, df=1):
    r"""Spatial NB likelihood-ratio test.

    ``LR = 2 (l_NB - l_Poisson)`` for Poisson against negative binomial
    (e.g. :func:`morie.fn.spcount.sar_poisson` against ``sar_negbin``). The
    null ``1 / theta = 0`` lies on the boundary of the parameter space, so
    ``LR`` is a 50:50 mixture of a point mass at 0 and chi-square(1) and the
    p-value is ``P(chi2_1 > LR) / 2`` (Self and Liang 1987; Cameron and
    Trivedi 2013, sec. 3.4). ``df`` other than 1 uses the plain chi-square.

    Parameters
    ----------
    ll_nb, ll_pois : float
        Log-likelihoods of the NB and the Poisson fit.
    df : int
        Number of dispersion parameters tested (1 for NB2).

    Returns
    -------
    SpatialResult
        ``statistic`` LR, ``p_value`` (boundary-corrected when ``df = 1``).

    References
    ----------
    Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and
    likelihood ratio tests under nonstandard conditions. *Journal of the American Statistical
    Association*, 82(398), 605-610.

    Cameron, A. C. and Trivedi, P. K. (2013). *Regression Analysis of Count Data*, 2nd ed. Cambridge
    University Press.

    Examples
    --------
    >>> r = scnblrt(-40.0, -42.0)
    >>> r.statistic, round(r.p_value, 12)
    (4.0, 0.022750131948)
    """
    lr = max(2.0 * (float(ll_nb) - float(ll_pois)), 0.0)
    if int(df) == 1:
        p = 0.5 * math.erfc(math.sqrt(lr / 2.0))
    else:
        from ._rrng_core import pchisq

        p = 1.0 - float(pchisq(lr, int(df)))
    return SpatialResult(name="scnblrt", statistic=lr, p_value=p, extra={"df": int(df)})


scnblrt_fn = scnblrt


def cheatsheet() -> str:
    return "scnblrt(ll_nb, ll_pois) -> boundary-corrected LR test of overdispersion"
