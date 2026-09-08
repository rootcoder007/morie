"""Deprecated alias for :func:`morie.fn.sdxbar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sdxbar import sdxbar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_93"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_93(sigma, N):
    """Deprecated; use :func:`morie.fn.sdxbar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_93() is the book-coordinate name for sdxbar(); "
        "it will be removed. Use morie.fn.sdxbar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(sigma, N)
