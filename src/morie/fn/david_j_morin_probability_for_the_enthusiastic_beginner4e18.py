"""Deprecated alias for :func:`morie.fn.poisson_small_interval`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisson_small_interval import poisson_small_interval as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_18"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_18(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.poisson_small_interval` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_18() is the book-coordinate name for poisson_small_interval(); "
        "it will be removed. Use morie.fn.poisson_small_interval() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
