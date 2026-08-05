"""Deprecated alias for :func:`morie.fn.sumdens`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sumdens import sumdens as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_65"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_65(grid_x, density_x, grid_y, density_y, z):
    """Deprecated; use :func:`morie.fn.sumdens` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_65() is the book-coordinate name for sumdens(); "
        "it will be removed. Use morie.fn.sumdens() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(grid_x, density_x, grid_y, density_y, z)
