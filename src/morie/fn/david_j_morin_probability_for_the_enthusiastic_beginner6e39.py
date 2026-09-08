"""Deprecated alias for :func:`morie.fn.groupavg`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .groupavg import groupavg as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_39"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_39(m, xavg):
    """Deprecated; use :func:`morie.fn.groupavg` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_39() is the book-coordinate name for groupavg(); "
        "it will be removed. Use morie.fn.groupavg() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(m, xavg)
