# morie.fn -- function file (rootcoder007/morie)
"""General copula frailty for clustered survival."""

import numpy as np

from ._copula import FAMILIES, copula_cdf, tau_to_theta
from ._richresult import RichResult
from .clyfr import _km

__all__ = ["copula_frailty"]


def copula_frailty(time1, event1, time2, event2, family="clayton", theta=None):
    r"""Survival copula for paired event times, any supported family.

    Applies the chosen copula to the Kaplan-Meier survival functions,

    .. math:: S(t_1, t_2) = C\big(S_1(t_1), S_2(t_2)\big),

    which is Sklar's theorem on the survival scale. Generalises
    :mod:`morie.fn.clyfr` beyond Clayton: Gumbel gives upper-tail
    (late-event) association, Clayton lower-tail (early-event),
    Frank neither -- the choice is a statement about *where* in time
    the pairing acts, not just how strong it is.

    Parameters
    ----------
    time1, time2, event1, event2 : array-like, shape (n,)
        Paired times and event indicators.
    family : str, default "clayton"
        Any family in :data:`morie.fn._copula.FAMILIES`.
    theta : float, optional
        Copula parameter; inverted from the sample Kendall's tau when
        omitted.

    Returns
    -------
    RichResult
        keys: ``family``, ``theta``, ``tau_sample``,
        ``joint_survival``, ``s1``, ``s2``, ``n_pairs``, ``method``.

    References
    ----------
    Sklar, A. (1959). Fonctions de repartition a n dimensions et
    leurs marges. *Publications de l'Institut de Statistique de
    l'Universite de Paris*, 8, 229-231.

    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Ch. 3, Table 3.2 p. 54.
    """
    from scipy import stats as _st

    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}.")
    t1 = np.asarray(time1, dtype=float).ravel()
    t2 = np.asarray(time2, dtype=float).ravel()
    e1 = np.asarray(event1, dtype=float).ravel()
    e2 = np.asarray(event2, dtype=float).ravel()
    n = t1.size
    if not (t2.size == n and e1.size == n and e2.size == n):
        raise ValueError("all four inputs must have the same length.")
    if n < 5:
        raise ValueError("need at least 5 pairs.")
    if np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("times must be positive.")

    tau_hat = float(_st.kendalltau(t1, t2).statistic)
    if theta is None and family != "independence":
        theta = tau_to_theta(family, tau_hat)
    g1, v1 = _km(t1, e1)
    g2, v2 = _km(t2, e2)
    s1 = np.maximum(v1[np.searchsorted(g1, t1, side="right") - 1], 1e-8)
    s2 = np.maximum(v2[np.searchsorted(g2, t2, side="right") - 1], 1e-8)
    joint = copula_cdf(family, s1, s2, theta)

    return RichResult(
        payload={
            "family": family,
            "theta": None if theta is None else float(theta),
            "tau_sample": tau_hat,
            "joint_survival": joint,
            "s1": s1,
            "s2": s2,
            "n_pairs": int(n),
            "method": f"Survival copula ({family}) on Kaplan-Meier margins",
        }
    )


def cheatsheet():
    return "copfr: S(t1,t2) = C(S1, S2), family choice sets WHERE the association acts"
