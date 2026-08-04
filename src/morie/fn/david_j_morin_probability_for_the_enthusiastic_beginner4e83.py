"""Deprecated alias for :func:`morie.fn.expmean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .expmean import expmean as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_83"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_83(tau):
    """Deprecated; use :func:`morie.fn.expmean` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_83() is the book-coordinate name for expmean(); "
        "it will be removed. Use morie.fn.expmean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(tau)
