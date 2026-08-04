"""Deprecated alias for :func:`morie.fn.varsum`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .varsum import varsum as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_25"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_25(var_x, var_y):
    """Deprecated; use :func:`morie.fn.varsum` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_25() is the book-coordinate name for varsum(); "
        "it will be removed. Use morie.fn.varsum() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl([var_x, var_y])
