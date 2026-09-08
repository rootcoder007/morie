"""Deprecated alias for :func:`morie.fn.powlogser`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .powlogser import powlogser as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_21"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_21(a, n, terms=12):
    """Deprecated; use :func:`morie.fn.powlogser` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_21() is the book-coordinate name for powlogser(); "
        "it will be removed. Use morie.fn.powlogser() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(a, n, terms)
