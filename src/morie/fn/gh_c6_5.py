# morie.fn -- function file (rootcoder007/morie)
"""Schwartz posterior-consistency conditions."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["kldsupp", "ghosal_schwartz_thm"]


def kldsupp(prior_mass, kl_radius, test_rate, n):
    """Check the extended Schwartz conditions and return the resulting bound.

    Two things must hold together and the theorem fails if either is
    dropped: the prior must put POSITIVE mass on a Kullback-Leibler
    neighbourhood of the truth, and the test must beat that
    neighbourhood's radius -- C > c, strictly.  Prior mass alone is not
    enough (there are famous inconsistent priors with full weak
    support), and a good test alone is not enough either.

    The margin C - c is returned because it is the quantity that
    governs the bound: the posterior mass of the alternative decays
    like e^{-(C-c)n}, so a margin near zero means no useful rate even
    though the conditions technically hold.

    Formula: Pi(P_0) > 0 and K(p_0; P_0) <= c and C > c imply
             Pi_n(P_n | X_1..X_n) -> 0 a.s.;
             the working bound is e^{-(C - c) n} / Pi(P_0)

    Parameters
    ----------
    prior_mass : float
        Pi(P_0), the prior mass of the Kullback-Leibler set, in (0, 1].
    kl_radius : float
        c, the Kullback-Leibler radius K(p_0; P_0).
    test_rate : float
        C, the exponential rate of the test (see ``gh_ap_d1``).
    n : int
        Sample size.

    Returns
    -------
    RichResult
        ``holds`` (1 if C > c and the mass is positive), ``margin``
        (C - c), ``bound``, ``prior_mass``, ``n``.

    References
    ----------
    Ghosal & van der Vaart (2017), Fundamentals of Nonparametric
    Bayesian Inference, Theorem 6.17 (Extended Schwartz): "If there
    exist a set P_0 subset P and number c with Pi(P_0) > 0 and
    K(p_0; P_0) <= c, then Pi_n(P_n | X_1, ..., X_n) -> 0 a.s. for any
    sets P_n such that either (a) or (b) holds for some constant
    C > c", with (b) Pi(P_n) <= e^{-Cn}; together with Theorem 6.16.
    Read from the copy of the book held in the corpus.
    """
    pm = float(prior_mass)
    c = float(kl_radius)
    Cc = float(test_rate)
    n = int(n)
    if not 0.0 < pm <= 1.0:
        raise ValueError("the prior mass must lie in (0, 1]")
    if c < 0.0:
        raise ValueError("the Kullback-Leibler radius must be non-negative")
    if n < 1:
        raise ValueError("n must be at least 1")
    margin = Cc - c
    return RichResult(payload={
        "holds": 1.0 if margin > 0.0 else 0.0, "margin": margin,
        "bound": math.exp(-margin * n) / pm, "prior_mass": pm,
        "n": float(n),
        "method": "Extended Schwartz conditions, Ghosal Theorem 6.17"})


ghosal_schwartz_thm = kldsupp


def cheatsheet():
    return "gh_c6_5: need Pi(P0)>0, K(p0;P0)<=c, C>c; bound e^-(C-c)n/Pi(P0)"
