"""Deprecated alias for :func:`morie.fn.exact_half_heads`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .exact_half_heads import exact_half_heads as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_65"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_65(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.exact_half_heads` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_65() is the book-coordinate name for exact_half_heads(); "
        "it will be removed. Use morie.fn.exact_half_heads() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
