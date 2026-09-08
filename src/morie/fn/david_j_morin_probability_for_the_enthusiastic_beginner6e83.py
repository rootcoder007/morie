"""Deprecated alias for :func:`morie.fn.bookmeans`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bookmeans import bookmeans as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_83"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_83(x=None, y=None):
    """Deprecated; use :func:`morie.fn.bookmeans` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_83() is the book-coordinate name for bookmeans(); "
        "it will be removed. Use morie.fn.bookmeans() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, y)
