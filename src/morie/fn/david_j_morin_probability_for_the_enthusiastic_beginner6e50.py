"""Deprecated alias for :func:`morie.fn.lsqrev`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .lsqrev import lsqrev as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_50"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_50(x, y):
    """Deprecated; use :func:`morie.fn.lsqrev` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_50() is the book-coordinate name for lsqrev(); "
        "it will be removed. Use morie.fn.lsqrev() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, y)
