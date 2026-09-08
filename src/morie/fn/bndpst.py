"""Post-specification-test bound -- NOT IMPLEMENTED"""

from ._richresult import RichResult  # noqa: F401  (kept for API stability)

__all__ = ["bound_post_test"]

_WHY = (
    'bound_post_test is not implemented.\n'
    '\n'
    'What was here before was a one-sample Kolmogorov-Smirnov test against\n'
    'a fitted normal, pasted in by a generator. It had nothing to do with\n'
    'constructing a confidence interval that stays valid after a\n'
    'specification test, but it ran and returned a plausible statistic and\n'
    'p-value, so a caller could not tell it was wrong. It has been deleted\n'
    'rather than left in place.\n'
    '\n'
    'It was not replaced because the construction could not be verified\n'
    'against a source. A confidence set that remains valid after a\n'
    'moment-inequality specification test is the generalized moment\n'
    'selection of Andrews & Soares (2010, Econometrica 78(1), 119-157):\n'
    'each inequality is classified as binding or slack by comparing the\n'
    'studentised sample moment against a tuning sequence kappa_n, and the\n'
    'critical value is recomputed from only the retained inequalities. The\n'
    'whole content of the method is in the selection function phi, the\n'
    'choice of kappa_n and how the critical value is simulated, none of\n'
    'which could be read off an accessible copy.\n'
    '\n'
    'Searched, without obtaining the definitions: the article on the\n'
    'publisher site and JSTOR (paywalled); ar5iv/arXiv for a restatement;\n'
    "and Molinari's Handbook of Econometrics chapter on partial\n"
    'identification (arXiv:2004.11751), which cites the method but does not\n'
    'print the selection function or the tuning sequence.\n'
    '\n'
    'To implement it, read Andrews & Soares (2010) Sections 3-4 for phi and\n'
    "kappa_n, or take them from an author's own code release.\n"
)


def bound_post_test(lower, upper, spec_test, cdf=None):
    """
    Post-specification-test bound -- NOT IMPLEMENTED

    Raises ``NotImplementedError``.  See the module note: the generator
    boilerplate that was here (a Kolmogorov-Smirnov test) was wrong and
    has been removed, and the Andrews-Soares generalized moment selection that should replace it could not be verified from any accessible source.

    References
    ----------
    Andrews D W K & Soares G (2010).  Inference for parameters defined by
    moment inequalities using generalized moment selection.  Econometrica
    78(1), 119-157.
    """
    raise NotImplementedError(_WHY)


def cheatsheet():
    return "bndpst: NOT IMPLEMENTED (Andrews-Soares GMS unverified)"


# compact alias per ledger/NAMING.md
boundposttest = bound_post_test
