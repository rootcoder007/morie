# morie.fn -- function file (rootcoder007/morie)
"""Exponential-consistency certificate for a test sequence."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["exptest", "ghosal_exp_test"]


def exptest(err_null, err_alt, n):
    """Certify that a test sequence has exponentially small error probabilities.

    Schwartz's theorem does not need tests whose errors merely go to
    zero -- it needs them to go to zero EXPONENTIALLY, and the rate
    constant C is what later has to beat the Kullback-Leibler radius.
    This turns two observed error probabilities at sample size n into
    that constant, so the comparison C > c can actually be made.

    The rate is read off a single n, so it is a certificate at that n
    and not a proof of a rate; a caller checking a sequence should call
    it at several n and see the constant stabilise.

    Formula: C_0 = -log(P_0^n phi_n)/n,  C_1 = -log(sup P^n(1 - phi_n))/n,
             C = min(C_0, C_1); the pair is exponentially consistent at
             rate C when both errors are <= e^{-Cn}

    Parameters
    ----------
    err_null : float
        Type I error P_0^n phi_n, in (0, 1].
    err_alt : float
        Worst-case type II error sup_{p in U^c} P^n (1 - phi_n), in (0, 1].
    n : int
        Sample size at which the errors were observed.

    Returns
    -------
    RichResult
        ``rate`` (C), ``rate_null``, ``rate_alt``, ``bound``
        (e^{-Cn}), ``exponential`` (1 if C > 0), ``n``.

    References
    ----------
    Ghosal & van der Vaart (2017), Fundamentals of Nonparametric
    Bayesian Inference, Theorem 6.16 (Schwartz) and its proof, which
    invokes Lemma D.11 to assume without loss of generality that "the
    tests phi_n ... have exponentially small error probabilities in the
    sense that P_0^n phi_n <= e^{-Cn} and sup_{p in U^c} P^n(1 - phi_n)
    <= e^{-Cn}, for some positive constant C".  Read from the copy of
    the book held in the corpus.
    """
    e0 = float(err_null)
    e1 = float(err_alt)
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 < e0 <= 1.0 or not 0.0 < e1 <= 1.0:
        raise ValueError("error probabilities must lie in (0, 1]")
    c0 = -math.log(e0) / n
    c1 = -math.log(e1) / n
    c = min(c0, c1)
    return RichResult(payload={
        "rate": c, "rate_null": c0, "rate_alt": c1,
        "bound": math.exp(-c * n), "exponential": 1.0 if c > 0 else 0.0,
        "n": float(n),
        "method": "Exponential test-consistency rate, Ghosal Theorem 6.16"})


ghosal_exp_test = exptest


def cheatsheet():
    return "gh_ap_d1: C = min(-log err0, -log err1)/n; need both <= e^-Cn"
