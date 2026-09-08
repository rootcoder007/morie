"""Deprecated alias for :func:`morie.fn.starbars`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .starbars import starbars as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_16"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_16(n, N):
    """Deprecated; use :func:`morie.fn.starbars` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_16() is the book-coordinate name for starbars(); "
        "it will be removed. Use morie.fn.starbars() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n, N)
