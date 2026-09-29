# morie.fn -- function file (rootcoder007/morie)
"""Gravity model negative-binomial PPML."""

from __future__ import annotations

from . import _gravity as gv
from ._containers import SpatialResult


def igravnb(flows, mass_o, mass_d, dist):
    r"""Gravity model negative-binomial PPML.

    The gravity equation ``E F = exp(b0 + b1 log M_o + b2 log M_d + b3 log
    d)`` fitted by negative-binomial (NB2, ``Var F = mu + mu^2 / theta``)
    maximum likelihood: Fisher scoring for ``b`` alternating with Newton
    steps on the ``theta`` score, exactly ``MASS::glm.nb`` (Cameron and
    Trivedi 2013, ch. 3). Used for overdispersed flows (Burger, van Oort and
    Linders 2009); model-based and HC0 standard errors are returned.

    Parameters
    ----------
    flows : array-like, shape (n,)
        Observed flows of the n origin-destination pairs.
    mass_o, mass_d : array-like, shape (n,)
        Positive origin and destination masses of each pair.
    dist : array-like, shape (n,)
        Positive distances.

    Returns
    -------
    SpatialResult
        ``statistic`` is the distance elasticity; ``extra`` has
        ``coefficients``, ``se``, ``se_robust``, ``theta``, ``loglik``,
        ``fitted``.

    References
    ----------
    Cameron, A. C. and Trivedi, P. K. (2013). *Regression Analysis of Count Data*, 2nd ed.
    Cambridge University Press, ch. 3.

    Burger, M., van Oort, F. and Linders, G.-J. (2009). On the specification of the gravity model
    of trade: zeros, excess zeros and zero-inflated estimation. *Spatial Economic Analysis*, 4(2),
    167-190.

    Examples
    --------
    >>> F = [12.0, 0.0, 30.0, 2.0, 85.0, 4.0, 19.0, 1.0, 7.0, 40.0, 3.0, 16.0]
    >>> mo = [5, 5, 9, 9, 20, 20, 7, 7, 12, 12, 3, 3]
    >>> md = [9, 20, 5, 20, 5, 9, 20, 9, 7, 3, 12, 5]
    >>> d = [1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 2.5, 1.5, 2.0, 1.0, 3.0, 1.2]
    >>> r = igravnb(F, mo, md, d)
    >>> round(r.statistic, 8), round(r.extra["theta"], 6)
    (0.29555178, 1.168603)
    """
    g = gv.glm_fit(gv.design(mass_o, mass_d, dist), gv.vec(flows), "negbin")
    return SpatialResult(
        name="igravnb",
        statistic=g["coefficients"][3],
        extra={k: g[k] for k in ("coefficients", "se", "se_robust", "theta", "loglik", "fitted")},
    )


igravnb_fn = igravnb


def cheatsheet() -> str:
    return "igravnb(flows, mass_o, mass_d, dist) -> NB2 gravity (MASS::glm.nb)"
