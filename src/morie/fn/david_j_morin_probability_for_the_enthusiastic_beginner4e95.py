"""Deprecated alias for :func:`morie.fn.poispeak`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poispeak import poispeak as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_95"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_95(n, p):
    """Deprecated; use :func:`morie.fn.poispeak` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_95() is the book-coordinate name for poispeak(); "
        "it will be removed. Use morie.fn.poispeak() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n, p)
