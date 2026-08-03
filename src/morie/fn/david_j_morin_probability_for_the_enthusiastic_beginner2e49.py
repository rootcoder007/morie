"""Deprecated alias for :func:`morie.fn.conditional_subset`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .conditional_subset import conditional_subset as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_49"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_49(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.conditional_subset` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_49() is the book-coordinate name for conditional_subset(); "
        "it will be removed. Use morie.fn.conditional_subset() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
