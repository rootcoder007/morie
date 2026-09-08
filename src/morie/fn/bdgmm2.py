# morie.fn -- function file (rootcoder007/morie)
"""Generalised moment selection test statistic."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["gmsbound", "bound_gmm_alt"]


def gmsbound(mbar, sigma, n, kappa=None):
    """Andrews-Soares moment selection for a moment-inequality test.

    Testing E m_j(theta) <= 0 for all j with a statistic that treats every
    inequality as binding is conservative: inequalities that are slack in
    the population contribute nothing in the limit but inflate the
    critical value.  Generalised moment selection drops them first.  The
    standardised moments and the selection statistic are

        t_j    = sqrt(n) mbar_j / sigma_j
        xi_j   = t_j / kappa_n,           kappa_n = sqrt(log n) by default

    and moment j is retained only when xi_j > -1, i.e. when it is not
    clearly slack; the test statistic over the retained set is the
    modified method of moments

        S = sum_{j retained} [ max(t_j, 0) ]^2.

    Parameters
    ----------
    mbar : array-like
        Sample moment means, length J, oriented so that positive means
        violate the inequality.
    sigma : array-like
        Moment standard deviations, length J, strictly positive.
    n : int
        Sample size.
    kappa : float or None
        Tuning sequence kappa_n; ``None`` uses sqrt(log n).

    Returns
    -------
    RichResult
        ``S``, ``t``, ``xi``, ``retained``, ``nretained``, ``kappa``,
        ``n``, ``J``.

    References
    ----------
    Andrews, D. W. K. and Soares, G. (2010), "Inference for parameters
    defined by moment inequalities using generalized moment selection",
    Econometrica 78(1), 119-157.  Their Sect. 4 defines xi_j as the
    standardised moment divided by a tuning sequence kappa_n, recommends
    kappa_n = sqrt(log n), and selects moments by whether xi_j exceeds a
    fixed threshold; the modified method of moments statistic
    S_1(m, Sigma) = sum_j [m_j]_+^2 is their Sect. 3.  Standard published
    form; the Econometrica article was not in the local corpus and was
    not read for this implementation.
    """
    m = C.vec(mbar)
    s = C.vec(sigma)
    J = len(m)
    if len(s) != J:
        raise ValueError("mbar and sigma must have the same length")
    if any(v <= 0.0 for v in s):
        raise ValueError("standard deviations must be strictly positive")
    n = float(n)
    if n <= 1.0:
        raise ValueError("n must exceed 1")
    k = math.sqrt(math.log(n)) if kappa is None else float(kappa)
    if k <= 0.0:
        raise ValueError("kappa must be strictly positive")
    t = [math.sqrt(n) * m[j] / s[j] for j in range(J)]
    xi = [t[j] / k for j in range(J)]
    keep = [1 if xi[j] > -1.0 else 0 for j in range(J)]
    S = sum(max(t[j], 0.0) ** 2 for j in range(J) if keep[j])
    return RichResult(payload={
        "S": S, "t": t, "xi": xi, "retained": keep,
        "nretained": sum(keep), "kappa": k, "n": n, "J": J,
        "method": "Generalised moment selection (Andrews-Soares 2010)"})


bound_gmm_alt = gmsbound


boundgmmalt = gmsbound


def cheatsheet():
    return "bdgmm2: Generalised moment selection test statistic."
