"""Deprecated alias for :func:`morie.fn.expintp`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .expintp import expintp as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_23"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_23(t, dt, lam):
    """Deprecated; use :func:`morie.fn.expintp` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_23() is the book-coordinate name for expintp(); "
        "it will be removed. Use morie.fn.expintp() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(t, dt, lam)
