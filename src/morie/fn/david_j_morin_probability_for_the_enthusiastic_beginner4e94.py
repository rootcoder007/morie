"""Deprecated alias for :func:`morie.fn.poisvar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisvar import poisvar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_94"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_94(a):
    """Deprecated; use :func:`morie.fn.poisvar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_94() is the book-coordinate name for poisvar(); "
        "it will be removed. Use morie.fn.poisvar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(a)
