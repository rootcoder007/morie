"""Deprecated alias for :func:`morie.fn.sdcoinsum`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sdcoinsum import sdcoinsum as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_51"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_51(n):
    """Deprecated; use :func:`morie.fn.sdcoinsum` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_51() is the book-coordinate name for sdcoinsum(); "
        "it will be removed. Use morie.fn.sdcoinsum() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n)
