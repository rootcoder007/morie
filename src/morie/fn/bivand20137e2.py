"""Extractor artefact -- NOT IMPLEMENTED"""

from ._richresult import RichResult  # noqa: F401  (kept for API stability)

__all__ = ["bivand2013_chapter_7_equation_2"]

_WHY = (
    'bivand2013_chapter_7_equation_2 is not implemented.\n'
    '\n'
    'What was here before was a one-sample Kolmogorov-Smirnov test against\n'
    'a fitted normal, pasted in by a generator. It ran and returned a\n'
    'plausible statistic and p-value, so a caller could not tell it was\n'
    'wrong. It has been deleted rather than left in place.\n'
    '\n'
    'It was not replaced because there is no method to implement. This\n'
    'module was produced by an equation extractor, and the text it captured\n'
    'as the specification is\n'
    '\n'
    "    'The values of the coefficients alpha, beta_1, ..., beta_5 are\n"
    "     5.56, 5.66, -0.963, -5.14'\n"
    '\n'
    'which is a sentence reporting fitted numbers in a worked example, not\n'
    'a formula and not a method. The module name is a book coordinate\n'
    '(chapter 7, equation 2) rather than a method name, which also violates\n'
    "ledger/NAMING.md: 'method names, never book coordinates'.\n"
    '\n'
    'This module should be deleted rather than implemented. It is one of\n'
    '281 rows in slice k03 with the same defect; see the slice report.\n'
)


def bivand2013_chapter_7_equation_2(x, cdf=None):
    """
    Extractor artefact -- NOT IMPLEMENTED

    Raises ``NotImplementedError``.  See the module note: the generator
    boilerplate that was here (a Kolmogorov-Smirnov test) was wrong and
    has been removed, and there is no method in the docstring to implement -- the extractor captured a sentence of prose, not a formula.

    References
    ----------
    None.  The docstring of this module names no method; see the note
    above.
    """
    raise NotImplementedError(_WHY)


def cheatsheet():
    return "bivand20137e2: NOT IMPLEMENTED (no method in the docstring)"
