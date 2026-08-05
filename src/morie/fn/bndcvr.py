# morie.fn -- function file (rootcoder007/morie)
"""Monte Carlo coverage check for a family of bound estimates."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_coverage_check"]


def bound_coverage_check(lower, upper, theta_true, alpha=0.05):
    """Empirical coverage of replicated intervals, tested against nominal.

    Coverage is the property a confidence construction claims and the one
    thing a simulation can check directly.  Under-coverage is the failure
    that matters, so the test is one-sided: the number of covering
    replications is binomial with the nominal success probability, and the
    reported p-value is its exact lower tail.  A two-sided test would
    flag conservative procedures, which are not wrong in the same way.

    Formula: ``coverage = (1 / R) sum 1{lower_r <= theta <= upper_r}``;
    ``p = P(Bin(R, 1 - alpha) <= n_covered)`` computed exactly.

    Parameters
    ----------
    lower, upper : array-like
        Replicated interval endpoints, same length ``R``.
    theta_true : float
        The data-generating parameter value.
    alpha : float, optional
        Nominal miss probability, default 0.05.

    Returns
    -------
    RichResult
        ``coverage``, ``nominal``, ``n_covered``, ``R``, ``p_value``,
        ``reject``, ``mean_width``.

    References
    ----------
    The coverage requirements distinguished here -- covering the
    identified set versus covering each parameter in it -- are equations
    (4.11) to (4.14) of Molinari, F. (2021), Microeconometrics with
    partial identification, Handbook of Econometrics 7A
    (arXiv:2004.11751 pp. 97-100).  Andrews, D. W. K. & Soares, G.
    (2010), Econometrica 78(1), 119-157, doi:10.3982/ECTA7502, is the
    stub's attribution and the source of the uniform-coverage criterion
    the check is applied to.
    """
    lo = C.vec(lower)
    hi = C.vec(upper)
    R = len(lo)
    if R == 0:
        raise ValueError("bound_coverage_check: lower is empty")
    if len(hi) != R:
        raise ValueError("bound_coverage_check: lower and upper must have the same length")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("bound_coverage_check: alpha must lie in (0, 1)")
    t = float(theta_true)
    k = 0
    w = 0.0
    for i in range(R):
        if hi[i] < lo[i]:
            raise ValueError("bound_coverage_check: upper is below lower at some replicate")
        w += hi[i] - lo[i]
        if lo[i] <= t and t <= hi[i]:
            k += 1
    p = 1.0 - a
    term = (1.0 - p) ** R
    tail = term
    for j in range(1, k + 1):
        term = term * p * (R - j + 1) / ((1.0 - p) * j)
        tail += term
    if tail > 1.0:
        tail = 1.0
    return RichResult(payload={
        "coverage": k / float(R), "nominal": p, "n_covered": k, "R": R,
        "p_value": tail, "reject": 1.0 if tail < a else 0.0,
        "mean_width": w / R,
        "method": "Coverage probability check"})


def cheatsheet():
    return "bndcvr: Coverage probability check"
