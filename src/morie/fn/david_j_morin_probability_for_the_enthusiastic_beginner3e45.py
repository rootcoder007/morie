"""Deprecated alias for :func:`morie.fn.sdiidsum`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sdiidsum import sdiidsum as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_45"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_45(sigma, n):
    """Deprecated; use :func:`morie.fn.sdiidsum` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_45() is the book-coordinate name for sdiidsum(); "
        "it will be removed. Use morie.fn.sdiidsum() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(sigma, n)
