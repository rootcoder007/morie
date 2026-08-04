"""Deprecated alias for :func:`morie.fn.densprobc`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .densprobc import densprobc as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4(grid, density, center, width):
    """Deprecated; use :func:`morie.fn.densprobc` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4() is the book-coordinate name for densprobc(); "
        "it will be removed. Use morie.fn.densprobc() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(grid, density, center, width)
