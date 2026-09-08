"""Deprecated alias for :func:`morie.fn.poissum1`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poissum1 import poissum1 as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_11"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_11(a, kmax=200):
    """Deprecated; use :func:`morie.fn.poissum1` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_11() is the book-coordinate name for poissum1(); "
        "it will be removed. Use morie.fn.poissum1() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(a, kmax)
