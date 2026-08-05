"""Deprecated alias for :func:`morie.fn.lsqresid`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .lsqresid import lsqresid as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_92"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_92(x, y):
    """Deprecated; use :func:`morie.fn.lsqresid` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_92() is the book-coordinate name for lsqresid(); "
        "it will be removed. Use morie.fn.lsqresid() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, y)
