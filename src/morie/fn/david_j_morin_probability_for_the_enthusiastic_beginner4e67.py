"""Deprecated alias for :func:`morie.fn.binomvar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binomvar import binomvar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_67"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_67(n, p):
    """Deprecated; use :func:`morie.fn.binomvar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_67() is the book-coordinate name for binomvar(); "
        "it will be removed. Use morie.fn.binomvar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n, p)
