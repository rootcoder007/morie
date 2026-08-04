"""Deprecated alias for :func:`morie.fn.sampmean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sampmean import sampmean as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54(x):
    """Deprecated; use :func:`morie.fn.sampmean` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54() is the book-coordinate name for sampmean(); "
        "it will be removed. Use morie.fn.sampmean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x)
