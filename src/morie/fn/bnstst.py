"""Inference on an interval-identified parameter -- not implemented."""

from ._richresult import RichResult  # noqa: F401  (kept for API stability)

__all__ = ["bound_test_inference"]

_WHY = (
    "bound_test_inference is not implemented.\n"
    "\n"
    "What was here before was a one-sample Kolmogorov-Smirnov test against\n"
    "a fitted normal, pasted in by a generator. It had nothing to do with\n"
    "inference on a partially identified parameter, but it ran and returned\n"
    "a plausible statistic and p-value, so a caller could not tell it was\n"
    "wrong. It has been deleted rather than left in place.\n"
    "\n"
    "It was not replaced because the construction could not be verified\n"
    "against a source. The test of H0: lower <= 0 <= upper for an\n"
    "interval-identified parameter turns on the critical value of\n"
    "Imbens & Manski (2004, Econometrica 72(6), 1845-1857) as corrected by\n"
    "Stoye (2009, Econometrica 77(4), 1299-1315), whose pre-test on the\n"
    "width of the estimated bounds decides between a one-sided and a\n"
    "two-sided critical value. Getting that pre-test threshold or the\n"
    "critical-value equation even slightly wrong yields a number that looks\n"
    "right and has the wrong coverage.\n"
    "\n"
    "Searched, without obtaining the defining equation: both articles on\n"
    "the publisher and JSTOR (paywalled, HTML error pages returned);\n"
    "the author's own page for the Stoye reprint (404); arXiv/ar5iv for\n"
    "restatements (1104.4630, 1601.03572, 2107.04785, 1911.01547);\n"
    "and Molinari's Handbook of Econometrics chapter on partial\n"
    "identification (arXiv:2004.11751), which describes Stoye's pre-test\n"
    "in words -- 'if the bounds are sufficiently close, expand by a\n"
    "two-sided critical value, otherwise by a one-sided' -- but never\n"
    "prints the threshold or the critical-value equation.\n"
    "\n"
    "To implement it, read Stoye (2009) equation for C_n and the\n"
    "accompanying pre-test, or take the construction from an author's\n"
    "own code release."
)


def bound_test_inference(lower, upper, se, cdf=None):
    """
    Test of an interval-identified parameter against zero -- NOT IMPLEMENTED

    Raises ``NotImplementedError``.  See the module note: the generator
    boilerplate that was here (a Kolmogorov-Smirnov test) was wrong and
    has been removed, and the Imbens-Manski / Stoye critical value that
    should replace it could not be verified from any accessible source.

    References
    ----------
    Imbens G W & Manski C F (2004).  Confidence intervals for partially
    identified parameters.  Econometrica 72(6), 1845-1857.

    Stoye J (2009).  More on confidence intervals for partially
    identified parameters.  Econometrica 77(4), 1299-1315.
    """
    raise NotImplementedError(_WHY)


def cheatsheet():
    return "bnstst: NOT IMPLEMENTED (Imbens-Manski/Stoye CI unverified)"

