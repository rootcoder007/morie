"""Deprecated alias for :func:`morie.fn.binomial_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binomial_variance import binomial_variance as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_33"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_33(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.binomial_variance` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_33() is the book-coordinate name for binomial_variance(); "
        "it will be removed. Use morie.fn.binomial_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
