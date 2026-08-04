"""Deprecated alias for :func:`morie.fn.poisstirc`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisstirc import poisstirc as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_17"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_17(x_dev, a):
    """Deprecated; use :func:`morie.fn.poisstirc` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_17() is the book-coordinate name for poisstirc(); "
        "it will be removed. Use morie.fn.poisstirc() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x_dev, a)
