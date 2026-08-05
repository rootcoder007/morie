# morie.fn -- function file (rootcoder007/morie)
"""Inference on an interval-identified parameter."""

from ._richresult import RichResult
from .bndfre import bound_frequentist

__all__ = ["bound_test_inference"]


def bound_test_inference(lower, upper, se=0.0, cdf=0.05):
    """Test a null value against the Imbens-Manski confidence interval.

    The decision is whether the interval that covers the true parameter
    with probability ``1 - alpha`` contains ``theta_0``.  The critical
    value is the one already implemented in
    :func:`~morie.fn.bndvar.bound_variance_term` and reached through
    :func:`~morie.fn.bndfre.bound_frequentist`; nothing about the
    construction is re-derived here.

    Parameters
    ----------
    lower, upper : array-like
        Replicated estimates of the lower and upper bound, same length.
    se : float, optional
        The null value ``theta_0``, default 0.0.
    cdf : float, optional
        The level ``alpha``, default 0.05.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``covers``, ``reject``, ``c``,
        ``theta_0``, ``n``.

    Notes
    -----
    The two parameter names ``se`` and ``cdf`` are inherited generator
    boilerplate and are kept only so existing positional calls do not
    break; they carry the meanings documented above.

    Stoye's (2009) refinement, which pre-tests the width of the estimated
    bounds and switches between a one- and a two-sided critical value, is
    NOT implemented: its pre-test threshold could not be verified against
    an accessible copy of the paper.  What is implemented is the
    Imbens-Manski (2004) equation (6) interval, whose critical value
    already interpolates between the two normal quantiles and attains
    both limits.

    References
    ----------
    Imbens, G. W. & Manski, C. F. (2004).  Confidence intervals for
    partially identified parameters.  Econometrica 72(6), 1845-1857,
    equation (6).  doi:10.1111/j.1468-0262.2004.00555.x.

    Stoye, J. (2009).  More on confidence intervals for partially
    identified parameters.  Econometrica 77(4), 1299-1315.
    doi:10.3982/ECTA7347 -- the refinement not implemented here.
    """
    a = float(cdf)
    r = bound_frequentist(lower, upper, a)
    t0 = float(se)
    lo = float(r["lower"])
    hi = float(r["upper"])
    covers = 1.0 if (lo <= t0 and t0 <= hi) else 0.0
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "covers": covers, "reject": 1.0 - covers, "c": float(r["c"]),
        "theta_0": t0, "n": int(r["n"]),
        "method": "Inference on an interval-identified parameter"})


def cheatsheet():
    return "bnstst: Imbens-Manski interval test of a null value"
