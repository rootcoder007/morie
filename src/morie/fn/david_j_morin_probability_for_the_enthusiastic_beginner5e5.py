"""Deprecated alias for :func:`morie.fn.binomctrf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binomctrf import binomctrf as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_5"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_5(x, n):
    """Deprecated; use :func:`morie.fn.binomctrf` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_5() is the book-coordinate name for binomctrf(); "
        "it will be removed. Use morie.fn.binomctrf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, n)
