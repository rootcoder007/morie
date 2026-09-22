"""Deprecated alias for :func:`morie.fn.dievar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .dievar import dievar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20(sides=6):
    """Deprecated; use :func:`morie.fn.dievar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20() is the book-coordinate name for dievar(); "
        "it will be removed. Use morie.fn.dievar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(sides)


def cheatsheet() -> str:
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e20: david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20(sides) -> Deprecated; use :func:`morie.fn.dievar` instead."
