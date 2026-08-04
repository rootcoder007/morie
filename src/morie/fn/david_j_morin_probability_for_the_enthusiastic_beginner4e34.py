"""Deprecated alias for :func:`morie.fn.binpoislim`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binpoislim import binpoislim as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_34"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_34(k, n, a):
    """Deprecated; use :func:`morie.fn.binpoislim` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_34() is the book-coordinate name for binpoislim(); "
        "it will be removed. Use morie.fn.binpoislim() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(k, n, a)
