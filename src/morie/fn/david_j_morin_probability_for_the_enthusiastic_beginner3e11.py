"""Deprecated alias for :func:`morie.fn.pmf_sum_convolution`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .pmf_sum_convolution import pmf_sum_convolution as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_11"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_11(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.pmf_sum_convolution` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_11() is the book-coordinate name for pmf_sum_convolution(); "
        "it will be removed. Use morie.fn.pmf_sum_convolution() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
