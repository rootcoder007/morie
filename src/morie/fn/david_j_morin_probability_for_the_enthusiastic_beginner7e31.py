"""Deprecated alias for :func:`morie.fn.diffquot2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .diffquot2 import diffquot2 as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_31"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_31(x, delta):
    """Deprecated; use :func:`morie.fn.diffquot2` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_31() is the book-coordinate name for diffquot2(); "
        "it will be removed. Use morie.fn.diffquot2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, delta)
