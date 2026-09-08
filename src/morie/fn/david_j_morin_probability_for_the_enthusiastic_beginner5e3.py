"""Deprecated alias for :func:`morie.fn.binomctr`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binomctr import binomctr as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_3"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_3(x, n):
    """Deprecated; use :func:`morie.fn.binomctr` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_3() is the book-coordinate name for binomctr(); "
        "it will be removed. Use morie.fn.binomctr() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, n)
