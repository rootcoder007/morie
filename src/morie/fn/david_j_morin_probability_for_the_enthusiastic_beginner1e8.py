"""Deprecated alias for :func:`morie.fn.binomcoef`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binomcoef import binomcoef as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_8"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_8(N, n):
    """Deprecated; use :func:`morie.fn.binomcoef` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_8() is the book-coordinate name for binomcoef(); "
        "it will be removed. Use morie.fn.binomcoef() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(N, n)
