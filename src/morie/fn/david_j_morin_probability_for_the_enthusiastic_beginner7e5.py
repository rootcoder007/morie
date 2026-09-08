"""Deprecated alias for :func:`morie.fn.gaussdom`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gaussdom import gaussdom as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_5"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_5(x, n):
    """Deprecated; use :func:`morie.fn.gaussdom` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_5() is the book-coordinate name for gaussdom(); "
        "it will be removed. Use morie.fn.gaussdom() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, n)
