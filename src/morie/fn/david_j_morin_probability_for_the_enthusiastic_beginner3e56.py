"""Deprecated alias for :func:`morie.fn.sdbinom`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sdbinom import sdbinom as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_56"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_56(n=10000, p=1.0 / 6.0):
    """Deprecated; use :func:`morie.fn.sdbinom` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_56() is the book-coordinate name for sdbinom(); "
        "it will be removed. Use morie.fn.sdbinom() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n, p)
