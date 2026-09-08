"""Deprecated alias for :func:`morie.fn.poispmf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poispmf import poispmf as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_40"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_40(k, a):
    """Deprecated; use :func:`morie.fn.poispmf` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_40() is the book-coordinate name for poispmf(); "
        "it will be removed. Use morie.fn.poispmf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(k, a)
