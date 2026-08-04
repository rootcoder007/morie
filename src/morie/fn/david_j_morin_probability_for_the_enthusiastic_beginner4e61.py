"""Deprecated alias for :func:`morie.fn.binommean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binommean import binommean as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_61"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_61(n, p):
    """Deprecated; use :func:`morie.fn.binommean` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_61() is the book-coordinate name for binommean(); "
        "it will be removed. Use morie.fn.binommean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n, p)
