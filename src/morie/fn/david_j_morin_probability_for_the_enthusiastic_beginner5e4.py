"""Deprecated alias for :func:`morie.fn.gaussapx`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gaussapx import gaussapx as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_4"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_4(x, n):
    """Deprecated; use :func:`morie.fn.gaussapx` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_4() is the book-coordinate name for gaussapx(); "
        "it will be removed. Use morie.fn.gaussapx() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, n)
