"""Deprecated alias for :func:`morie.fn.poismean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poismean import poismean as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_92"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_92(a):
    """Deprecated; use :func:`morie.fn.poismean` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_92() is the book-coordinate name for poismean(); "
        "it will be removed. Use morie.fn.poismean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(a)
