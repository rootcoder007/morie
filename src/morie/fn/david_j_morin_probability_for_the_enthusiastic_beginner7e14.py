"""Deprecated alias for :func:`morie.fn.powexpapx`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .powexpapx import powexpapx as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_14"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_14(a, n):
    """Deprecated; use :func:`morie.fn.powexpapx` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_14() is the book-coordinate name for powexpapx(); "
        "it will be removed. Use morie.fn.powexpapx() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(a, n)
