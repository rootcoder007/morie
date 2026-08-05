"""Deprecated alias for :func:`morie.fn.lsqfit`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .lsqfit import lsqfit as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_47"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_47(x, y):
    """Deprecated; use :func:`morie.fn.lsqfit` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_47() is the book-coordinate name for lsqfit(); "
        "it will be removed. Use morie.fn.lsqfit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, y)
