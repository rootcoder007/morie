# morie.fn -- function file (rootcoder007/morie)
"""L-moment estimator of the GEV parameters."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ev_gev_lmoments", "evt_gev_lmoments"]


def ev_gev_lmoments(block_maxima):
    r"""Hosking's (1990) L-moment fit of the generalised extreme-value
    distribution to block maxima.

    The sample L-moments come from the UNBIASED probability-weighted
    moments (not plotting positions), the shape from Hosking's
    approximation

    .. math:: \hat k \approx 7.8590\,c + 2.9554\,c^2, \qquad
              c = \frac2{3+t_3} - \frac{\log 2}{\log 3},

    and scale and location exactly from
    :math:`\alpha = l_2\hat k/((1-2^{-\hat k})\Gamma(1+\hat k))`,
    :math:`\mu = l_1 - \alpha(1-\Gamma(1+\hat k))/\hat k`.

    **Sign convention, stated because it burns people:** Hosking's
    :math:`k` is MINUS the extreme-value index :math:`\xi`. A heavy
    (Frechet) tail has :math:`k < 0` and :math:`\xi > 0`; conflating
    them turns every Frechet into a Weibull. Both are returned, named.

    Why L-moments and not maximum likelihood: the GEV's ML estimator
    misbehaves for :math:`\xi < -0.5` (non-regular likelihood) and is
    outperformed by L-moments in small samples across most of the
    parameter space Hosking studied -- and block-maxima samples ARE
    small, since each observation costs a block.

    Parameters
    ----------
    block_maxima : array-like
        One maximum per block.

    Returns
    -------
    RichResult
        keys: ``mu``, ``sigma``, ``k_hosking``, ``xi``,
        ``l1``, ``l2``, ``t3``, ``tail_type``, ``return_level_fn``
        (a callable T -> level), ``n_blocks``, ``method``.

    References
    ----------
    Hosking, J. R. M. (1990), "L-moments: analysis and estimation of
    distributions using linear combinations of order statistics",
    *JRSS-B* 52:105-124. Hosking, Wallis and Wood (1985),
    *Technometrics* 27:251-261, for the GEV specifics.
    """
    from ._evt import gev_from_lmoments, l_moments

    xv = np.asarray(block_maxima, dtype=float).ravel()
    n = xv.size
    if n < 10:
        raise ValueError(f"need at least 10 block maxima, got {n}.")
    l1, l2, l3, t3 = l_moments(xv)
    mu, sigma, k = gev_from_lmoments(l1, l2, t3)
    xi = -k
    tail = ("Frechet (heavy, xi > 0)" if xi > 0.01 else
            "Weibull (bounded, xi < 0)" if xi < -0.01 else
            "Gumbel (light, xi ~ 0)")

    def return_level(T):
        T = np.asarray(T, dtype=float)
        y = -np.log(1.0 - 1.0 / T)
        if abs(k) < 1e-9:
            return mu - sigma * np.log(y)
        return mu + sigma / k * (1.0 - y ** k)

    return RichResult(payload={
        "mu": mu, "sigma": sigma, "k_hosking": k, "xi": xi,
        "l1": l1, "l2": l2, "t3": t3,
        "tail_type": tail,
        "sign_convention": "Hosking's k = -xi: heavy tail means k < 0, "
                           "xi > 0",
        "return_level_fn": return_level,
        "why_not_ml": "GEV maximum likelihood is non-regular for xi < -0.5 "
                      "and loses to L-moments in the small samples block "
                      "maxima produce",
        "n_blocks": int(n),
        "method": "GEV by L-moments (Hosking 1990), unbiased PWMs"})


def cheatsheet():
    return "evgevlm: Hosking's k = -xi -- and L-moments beat ML in small block-maxima samples"


#: Catalogue alias for :func:`ev_gev_lmoments`.
evt_gev_lmoments = ev_gev_lmoments
