# morie.fn -- function file (rootcoder007/morie)
"""Bayes factor and posterior probability for a point null."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ptnulltst", "ghosal_pt_null_tst"]


def ptnulltst(loglik_null, log_marginal_alt, lam=0.5):
    """Bayes factor for H0: p = p* against a nonparametric alternative.

    Everything is done in LOGS, which is the only way this computation
    survives realistic sample sizes: the two likelihoods differ by
    hundreds of nats and forming their ratio directly overflows long
    before n is interesting.

    Note the weighting convention, which is the book's and is easy to
    invert by accident: lambda is the prior weight of the ALTERNATIVE,
    so the null carries 1 - lambda in Pi = (1 - lambda) delta_{p*} +
    lambda Pi_1.

    Formula: B_n = prod_i p*(X_i) / int prod_i p(X_i) dPi_1(p);
             Pi_n(p = p* | X) = (1-lam) prod p*(X_i)
                / [ (1-lam) prod p*(X_i) + lam int prod p(X_i) dPi_1(p) ]

    Parameters
    ----------
    loglik_null : float
        log prod_i p*(X_i).
    log_marginal_alt : float
        log int prod_i p(X_i) dPi_1(p).
    lam : float
        Prior weight of the ALTERNATIVE, in (0, 1).

    Returns
    -------
    RichResult
        ``log_bayes_factor``, ``bayes_factor`` (inf when it overflows),
        ``posterior_null``, ``posterior_alt``, ``prior_null``,
        ``lam``.

    References
    ----------
    Ghosal & van der Vaart (2017), Fundamentals of Nonparametric
    Bayesian Inference, Section 10.5.1 (Testing a Point Null): "If
    lambda in (0, 1) is the prior weight of the null model and Pi_1 is
    the prior distribution on p under the alternative model, then the
    overall prior is the mixture Pi = (1 - lambda) delta_{p*} + lambda
    Pi_1", with the displayed posterior probability and Bayes factor
    B_n, and Theorem 10.24 for their consistency.  The book's prose
    calls lambda the weight of the null while its own formula gives the
    null the weight 1 - lambda; the FORMULA is followed here, and the
    argument is documented as the weight of the alternative to remove
    the ambiguity.  Read from the copy of the book held in the corpus.
    """
    ln = float(loglik_null)
    la = float(log_marginal_alt)
    lam = float(lam)
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must lie strictly between 0 and 1")
    lbf = ln - la
    a = math.log(1.0 - lam) + ln
    b = math.log(lam) + la
    mx = max(a, b)
    denom = mx + math.log(math.exp(a - mx) + math.exp(b - mx))
    post0 = math.exp(a - denom)
    try:
        bf = math.exp(lbf)
    except OverflowError:
        bf = math.inf
    return RichResult(payload={
        "log_bayes_factor": lbf, "bayes_factor": bf,
        "posterior_null": post0, "posterior_alt": 1.0 - post0,
        "prior_null": 1.0 - lam, "lam": lam,
        "method": "Point-null Bayes factor, Ghosal Section 10.5.1"})


ghosal_pt_null_tst = ptnulltst


def cheatsheet():
    return "gh_c10_13: B_n = L(p*)/int L dPi_1; null carries prior 1 - lam"
