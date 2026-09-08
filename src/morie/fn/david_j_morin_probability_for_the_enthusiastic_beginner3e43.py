"""Deprecated alias for :func:`morie.fn.sdsum`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sdsum import sdsum as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_43"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_43(sigmas):
    """Deprecated; use :func:`morie.fn.sdsum` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_43() is the book-coordinate name for sdsum(); "
        "it will be removed. Use morie.fn.sdsum() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(sigmas)
