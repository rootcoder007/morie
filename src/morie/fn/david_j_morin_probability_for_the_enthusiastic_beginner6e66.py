"""Deprecated alias for :func:`morie.fn.sumdensp`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sumdensp import sumdensp as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_66"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_66(grid_x, density_x, grid_y, density_y, z, dz):
    """Deprecated; use :func:`morie.fn.sumdensp` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_66() is the book-coordinate name for sumdensp(); "
        "it will be removed. Use morie.fn.sumdensp() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(grid_x, density_x, grid_y, density_y, z, dz)
