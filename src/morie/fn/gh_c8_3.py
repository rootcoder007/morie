# morie.fn -- function file (rootcoder007/morie)
"""Contraction-rate conditions of the basic rate theorem."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["testcond", "ghosal_test_cond"]


def testcond(prior_ball, log_entropy, sieve_mass, eps_bar, eps, n, Cconst):
    """Check the three conditions that deliver a posterior contraction rate.

    The three conditions pull in different directions and the rate is
    whatever satisfies all of them at once: the prior must put ENOUGH
    mass near the truth, the sieve must be SMALL enough to be tested
    over, and everything outside the sieve must have NEGLIGIBLE prior
    mass.  Each is reported separately with its slack, because the
    binding one is what a caller has to fix.

    Note the asymmetry that is easy to miss: the prior-mass and
    sieve-mass conditions use epsilon-bar, the entropy condition uses
    epsilon, and the theorem also needs n epsilon-bar^2 -> infinity.

    Formula: (i)   Pi_n(B_2(p_0, ebar_n)) >= e^{-C n ebar_n^2}
             (ii)  log N(xi eps_n, P_{n,1}, d) <= n eps_n^2
             (iii) Pi_n(P_{n,2}) <= e^{-(C + 4) n ebar_n^2}

    Parameters
    ----------
    prior_ball : float
        Pi_n(B_2(p_0, ebar_n)), in (0, 1].
    log_entropy : float
        log N(xi eps_n, P_{n,1}, d), non-negative.
    sieve_mass : float
        Pi_n(P_{n,2}), in [0, 1].
    eps_bar : float
        ebar_n, positive.
    eps : float
        eps_n, positive and at least eps_bar.
    n : int
        Sample size.
    Cconst : float
        The constant C > 0.

    Returns
    -------
    RichResult
        ``holds``, ``cond_prior``, ``cond_entropy``, ``cond_sieve``,
        ``slack_prior``, ``slack_entropy``, ``slack_sieve``,
        ``n_eps_bar_sq``.

    References
    ----------
    Ghosal & van der Vaart (2017), Fundamentals of Nonparametric
    Bayesian Inference, Theorem 8.9 (Basic contraction rate), Section
    8.2, conditions (8.4), (8.5) and (8.6), with "constants ebar_n <=
    eps_n with n ebar_n^2 -> infinity".  Read from the copy of the book
    held in the corpus.
    """
    pb = float(prior_ball)
    le = float(log_entropy)
    sm = float(sieve_mass)
    eb = float(eps_bar)
    ep = float(eps)
    n = int(n)
    Cc = float(Cconst)
    if not 0.0 < pb <= 1.0:
        raise ValueError("the prior ball mass must lie in (0, 1]")
    if le < 0.0:
        raise ValueError("the log entropy must be non-negative")
    if not 0.0 <= sm <= 1.0:
        raise ValueError("the sieve mass must lie in [0, 1]")
    if eb <= 0.0 or ep <= 0.0:
        raise ValueError("the rates must be positive")
    if ep < eb:
        raise ValueError("eps_n must be at least eps_bar_n")
    if n < 1:
        raise ValueError("n must be at least 1")
    if Cc <= 0.0:
        raise ValueError("C must be positive")
    neb = n * eb * eb
    s1 = math.log(pb) + Cc * neb
    s2 = n * ep * ep - le
    s3 = -(Cc + 4.0) * neb - math.log(sm) if sm > 0.0 else math.inf
    c1 = 1.0 if s1 >= 0.0 else 0.0
    c2 = 1.0 if s2 >= 0.0 else 0.0
    c3 = 1.0 if s3 >= 0.0 else 0.0
    return RichResult(payload={
        "holds": 1.0 if c1 and c2 and c3 else 0.0, "cond_prior": c1,
        "cond_entropy": c2, "cond_sieve": c3, "slack_prior": s1,
        "slack_entropy": s2, "slack_sieve": s3, "n_eps_bar_sq": neb,
        "method": "Contraction-rate conditions, Ghosal Theorem 8.9"})


ghosal_test_cond = testcond


def cheatsheet():
    return "gh_c8_3: (8.4) prior mass, (8.5) entropy, (8.6) sieve mass"
