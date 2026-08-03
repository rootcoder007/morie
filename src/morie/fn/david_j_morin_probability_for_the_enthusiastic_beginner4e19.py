"""Deprecated alias for :func:`morie.fn.poisson_mean_rate`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisson_mean_rate import poisson_mean_rate as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_19"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_19(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.poisson_mean_rate` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_19() is the book-coordinate name for poisson_mean_rate(); "
        "it will be removed. Use morie.fn.poisson_mean_rate() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
