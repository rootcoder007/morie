"""Deprecated alias for :func:`morie.fn.covzmean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .covzmean import covzmean as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_8"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_8(x, y):
    """Deprecated; use :func:`morie.fn.covzmean` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_8() is the book-coordinate name for covzmean(); "
        "it will be removed. Use morie.fn.covzmean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, y)
