"""Deprecated alias for :func:`morie.fn.binomdie`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binomdie import binomdie as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_32"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_32(k, n, b):
    """Deprecated; use :func:`morie.fn.binomdie` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_32() is the book-coordinate name for binomdie(); "
        "it will be removed. Use morie.fn.binomdie() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(k, n, b)
