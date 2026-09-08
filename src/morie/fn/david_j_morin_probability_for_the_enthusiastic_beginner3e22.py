"""Deprecated alias for :func:`morie.fn.bernoulli_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bernoulli_variance import bernoulli_variance as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_22"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_22(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.bernoulli_variance` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_22() is the book-coordinate name for bernoulli_variance(); "
        "it will be removed. Use morie.fn.bernoulli_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
