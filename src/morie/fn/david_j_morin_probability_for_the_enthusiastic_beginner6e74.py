"""Deprecated alias for :func:`morie.fn.stripmean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .stripmean import stripmean as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_74"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_74(r, sigma_x, sigma_y, y0):
    """Deprecated; use :func:`morie.fn.stripmean` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_74() is the book-coordinate name for stripmean(); "
        "it will be removed. Use morie.fn.stripmean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(r, sigma_x, sigma_y, y0)
