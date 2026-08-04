"""Test of the monotonicity assumption -- NOT IMPLEMENTED"""

from ._richresult import RichResult  # noqa: F401  (kept for API stability)

__all__ = ["bound_monotone_test"]

_WHY = (
    'bound_monotone_test is not implemented.\n'
    '\n'
    'What was here before was a one-sample Kolmogorov-Smirnov test against\n'
    'a fitted normal, pasted in by a generator. It had nothing to do with\n'
    'testing monotonicity of a potential outcome in the treatment, but it\n'
    'ran and returned a plausible statistic and p-value, so a caller could\n'
    'not tell it was wrong. It has been deleted rather than left in place.\n'
    '\n'
    'It was not replaced because the construction could not be verified\n'
    'against a source. The testable implication of instrument validity and\n'
    'monotonicity is the pair of density inequalities of Kitagawa (2015,\n'
    'Econometrica 83(5), 2043-2063): for every measurable set B,\n'
    'P(Y in B, D = 1 | Z = 1) >= P(Y in B, D = 1 | Z = 0) and\n'
    'P(Y in B, D = 0 | Z = 0) >= P(Y in B, D = 0 | Z = 1). Turning that\n'
    'into a test requires the variance-weighted supremum statistic over a\n'
    'class of intervals, the bootstrap recentring and the tuning constant,\n'
    'which could not be read off an accessible copy. A test built on a\n'
    'guess at any of those would report a size that is not the nominal one.\n'
    '\n'
    'Searched, without obtaining the statistic: the article on the\n'
    'publisher site and JSTOR (paywalled); ar5iv/arXiv for a restatement;\n'
    "and Molinari's Handbook of Econometrics chapter on partial\n"
    'identification (arXiv:2004.11751), which cites Kitagawa but does not\n'
    'print the statistic or the bootstrap.\n'
    '\n'
    "Note also that this function's signature (y, D, X) has no instrument\n"
    'argument, so it could not express the test even if the statistic were\n'
    'in hand; the instrument would have to be added.\n'
    '\n'
    'To implement it, read Kitagawa (2015) Sections 3-4, or take the\n'
    "construction from the author's own code release.\n"
)


def bound_monotone_test(y, D, X, cdf=None):
    """
    Test of the monotonicity assumption -- NOT IMPLEMENTED

    Raises ``NotImplementedError``.  See the module note: the generator
    boilerplate that was here (a Kolmogorov-Smirnov test) was wrong and
    has been removed, and the Kitagawa instrument-validity statistic that should replace it could not be verified from any accessible source.

    References
    ----------
    Kitagawa T (2015).  A test for instrument validity.  Econometrica
    83(5), 2043-2063.
    """
    raise NotImplementedError(_WHY)


def cheatsheet():
    return "bndmnt: NOT IMPLEMENTED (Kitagawa validity test unverified)"

