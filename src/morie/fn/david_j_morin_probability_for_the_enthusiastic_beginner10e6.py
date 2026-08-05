"""Deprecated alias for :func:`morie.fn.linmodel`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .linmodel import linmodel as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_10_equation_6"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_10_equation_6(m=1.0, sigma_x=7.5, sigma_z=10.6):
    """Deprecated; use :func:`morie.fn.linmodel` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_10_equation_6() is the book-coordinate name for linmodel(); "
        "it will be removed. Use morie.fn.linmodel() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(m, sigma_x, sigma_z)
