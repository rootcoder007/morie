"""Deprecated alias for :func:`morie.fn.ordsubs`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ordsubs import ordsubs as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6(N, n):
    """Deprecated; use :func:`morie.fn.ordsubs` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6() is the book-coordinate name for ordsubs(); "
        "it will be removed. Use morie.fn.ordsubs() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(N, n)
