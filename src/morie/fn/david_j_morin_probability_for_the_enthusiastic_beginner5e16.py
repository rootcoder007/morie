"""Deprecated alias for :func:`morie.fn.poisstirl`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisstirl import poisstirl as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_16"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_16(k, a):
    """Deprecated; use :func:`morie.fn.poisstirl` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_16() is the book-coordinate name for poisstirl(); "
        "it will be removed. Use morie.fn.poisstirl() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(k, a)
