"""Deprecated alias for :func:`morie.fn.binompeak`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binompeak import binompeak as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_96"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_96(n, p):
    """Deprecated; use :func:`morie.fn.binompeak` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_96() is the book-coordinate name for binompeak(); "
        "it will be removed. Use morie.fn.binompeak() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n, p)
