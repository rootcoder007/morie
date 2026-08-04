"""Deprecated alias for :func:`morie.fn.binomexp`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binomexp import binomexp as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_35"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_35(x, n, delta):
    """Deprecated; use :func:`morie.fn.binomexp` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_35() is the book-coordinate name for binomexp(); "
        "it will be removed. Use morie.fn.binomexp() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, n, delta)
