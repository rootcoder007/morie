"""Deprecated alias for :func:`morie.fn.poiszero`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poiszero import poiszero as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_99"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_99(a=7.0):
    """Deprecated; use :func:`morie.fn.poiszero` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_99() is the book-coordinate name for poiszero(); "
        "it will be removed. Use morie.fn.poiszero() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(a)
