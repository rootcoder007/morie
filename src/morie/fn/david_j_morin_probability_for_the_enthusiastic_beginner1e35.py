"""Deprecated alias for :func:`morie.fn.multinom`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .multinom import multinom as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_35"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_35(ns, N=None):
    """Deprecated; use :func:`morie.fn.multinom` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_35() is the book-coordinate name for multinom(); "
        "it will be removed. Use morie.fn.multinom() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(ns, N)
