# morie.fn -- function file (rootcoder007/morie)
"""Prevalence ratio."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['prevratio', 'prevalence_ratio']


def prevratio(prev_exposed, prev_unexposed, n_exposed=None, n_unexposed=None, alpha=0.05):
    """Prevalence ratio.

    For a common outcome the odds ratio overstates the prevalence ratio, sometimes badly, which is the whole reason the prevalence ratio is preferred in cross-sectional work. The confidence interval is built on the log scale and only when group sizes are supplied -- a ratio with no denominators has no standard error, and returning one anyway would be an invention.


    Formula: PR = p_e / p_u; se(log PR) = sqrt((1-p_e)/(p_e n_e) + (1-p_u)/(p_u n_u))

    Parameters
    ----------
    prev_exposed : float
        Prevalence in the exposed group.
    prev_unexposed : float
        Prevalence in the unexposed group.
    n_exposed : int, optional
        Size of the exposed group.
    n_unexposed : int, optional
        Size of the unexposed group.
    alpha : float
        Two-sided significance level.

    Returns
    -------
    RichResult
        ``pr``, ``log_pr``, ``se_log``, ``ci_lower``, ``ci_upper``.

    References
    ----------
    Barros and Hirakata (2003), Alternatives for logistic regression in
    cross-sectional studies: an empirical comparison of models that
    directly estimate the prevalence ratio, BMC Medical Research
    Methodology 3:21.  Open access; the delta-method standard error for
    log PR used here is the standard binomial one.
    """
    pe = float(prev_exposed); pu = float(prev_unexposed)
    if not 0 < pe < 1 or not 0 < pu < 1:
        raise ValueError("prevalences must be strictly between 0 and 1")
    pr = pe / pu
    lo = hi = se = float("nan")
    if n_exposed is not None and n_unexposed is not None:
        se = math.sqrt((1 - pe) / (pe * float(n_exposed))
                       + (1 - pu) / (pu * float(n_unexposed)))
        z = C.qnorm(1.0 - float(alpha) / 2.0)
        lo = math.exp(math.log(pr) - z * se)
        hi = math.exp(math.log(pr) + z * se)
    return RichResult(payload={
        "pr": pr, "log_pr": math.log(pr), "se_log": se,
        "ci_lower": lo, "ci_upper": hi, "method": "Prevalence ratio"})


prevalence_ratio = prevratio


def cheatsheet():
    return "prratio: Prevalence ratio."
