"""Deprecated alias for :func:`morie.fn.sampr`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sampr import sampr as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_9(x, y):
    """Deprecated; use :func:`morie.fn.sampr` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_9() is the book-coordinate name for sampr(); "
        "it will be removed. Use morie.fn.sampr() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, y)
