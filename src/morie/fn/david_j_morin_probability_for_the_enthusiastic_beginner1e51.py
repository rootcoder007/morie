"""Deprecated alias for :func:`morie.fn.starbrec`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .starbrec import starbrec as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_51"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_51(n, N):
    """Deprecated; use :func:`morie.fn.starbrec` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_51() is the book-coordinate name for starbrec(); "
        "it will be removed. Use morie.fn.starbrec() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n, N)
