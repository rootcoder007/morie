"""Deprecated alias for :func:`morie.fn.varxbar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .varxbar import varxbar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_92"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_92(sigma, N):
    """Deprecated; use :func:`morie.fn.varxbar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_92() is the book-coordinate name for varxbar(); "
        "it will be removed. Use morie.fn.varxbar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(sigma, N)
