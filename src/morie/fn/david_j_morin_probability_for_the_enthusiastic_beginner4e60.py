"""Deprecated alias for :func:`morie.fn.binompmf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binompmf import binompmf as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_60"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_60(k, n, p):
    """Deprecated; use :func:`morie.fn.binompmf` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_60() is the book-coordinate name for binompmf(); "
        "it will be removed. Use morie.fn.binompmf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(k, n, p)
