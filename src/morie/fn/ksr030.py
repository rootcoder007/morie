# morie.fn -- function file (rootcoder007/morie)
"""Brownian bridge covariance."""

from . import _array_core as np

from ._kosorok import bridge_cov
from ._richresult import RichResult

__all__ = ["kosorok_ch2_brownian_bridge_covariance"]


def kosorok_ch2_brownian_bridge_covariance(s, t, F=None):
    r"""Covariance of the limiting Brownian bridge (PDF-verified,
    Kosorok Ch. 2):

    .. math:: \mathrm{cov}[G(s), G(t)] = F(s \wedge t) - F(s)F(t).

    With F the uniform CDF this is the standard bridge covariance
    :math:`s \wedge t - st`, which vanishes at both endpoints -- the
    defining "tied down" property that distinguishes the bridge from
    Brownian motion.

    Parameters
    ----------
    s, t : float or array-like
        Time points.
    F : callable, optional
        The CDF; uniform on [0, 1] if omitted.

    Returns
    -------
    RichResult
        keys: ``covariance``, ``variance_s`` (the s = t case),
        ``s``, ``t``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2.
    """
    cov = bridge_cov(s, t, F)
    var = bridge_cov(s, s, F)
    return RichResult(
        payload={"covariance": cov, "variance_s": var, "s": s, "t": t,
                 "method": "cov[G(s), G(t)] = F(s ^ t) - F(s)F(t) (Kosorok Ch. 2)"}
    )


def cheatsheet():
    return "ksr030: F(s^t) - F(s)F(t); zero at both endpoints"
