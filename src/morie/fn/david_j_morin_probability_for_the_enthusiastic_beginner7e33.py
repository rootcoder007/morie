"""Deprecated alias for :func:`morie.fn.diffquotn`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .diffquotn import diffquotn as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_33"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_33(x, n, delta):
    """Deprecated; use :func:`morie.fn.diffquotn` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_33() is the book-coordinate name for diffquotn(); "
        "it will be removed. Use morie.fn.diffquotn() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, n, delta)
