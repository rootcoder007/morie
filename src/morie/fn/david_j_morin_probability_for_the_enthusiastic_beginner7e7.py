"""Deprecated alias for :func:`morie.fn.exptaylor`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .exptaylor import exptaylor as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_7"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_7(x, terms=30):
    """Deprecated; use :func:`morie.fn.exptaylor` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_7() is the book-coordinate name for exptaylor(); "
        "it will be removed. Use morie.fn.exptaylor() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, terms)
