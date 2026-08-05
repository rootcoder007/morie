"""Deprecated alias for :func:`morie.fn.bestconst`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bestconst import bestconst as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_23"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_23(y):
    """Deprecated; use :func:`morie.fn.bestconst` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_23() is the book-coordinate name for bestconst(); "
        "it will be removed. Use morie.fn.bestconst() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(y)
