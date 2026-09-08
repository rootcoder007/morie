"""Deprecated alias for :func:`morie.fn.powexpap2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .powexpap2 import powexpap2 as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_24"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_24(a, n):
    """Deprecated; use :func:`morie.fn.powexpap2` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_24() is the book-coordinate name for powexpap2(); "
        "it will be removed. Use morie.fn.powexpap2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(a, n)
