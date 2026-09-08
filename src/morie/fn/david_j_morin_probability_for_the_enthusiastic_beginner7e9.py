"""Deprecated alias for :func:`morie.fn.explinapx`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .explinapx import explinapx as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_9(x):
    """Deprecated; use :func:`morie.fn.explinapx` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_9() is the book-coordinate name for explinapx(); "
        "it will be removed. Use morie.fn.explinapx() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x)
