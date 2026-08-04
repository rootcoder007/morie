"""Deprecated alias for :func:`morie.fn.halfheads`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .halfheads import halfheads as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_66"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_66(n):
    """Deprecated; use :func:`morie.fn.halfheads` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_66() is the book-coordinate name for halfheads(); "
        "it will be removed. Use morie.fn.halfheads() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n)
