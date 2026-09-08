"""Deprecated alias for :func:`morie.fn.pandind`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .pandind import pandind as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_70"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_70(p_a, p_b):
    """Deprecated; use :func:`morie.fn.pandind` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_70() is the book-coordinate name for pandind(); "
        "it will be removed. Use morie.fn.pandind() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl([p_a, p_b])
