"""Deprecated alias for :func:`morie.fn.covshort`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .covshort import covshort as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14(x, y):
    """Deprecated; use :func:`morie.fn.covshort` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14() is the book-coordinate name for covshort(); "
        "it will be removed. Use morie.fn.covshort() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, y)
