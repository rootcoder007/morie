"""Deprecated alias for :func:`morie.fn.poisson_zero_series`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisson_zero_series import poisson_zero_series as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_53"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_53(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.poisson_zero_series` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_53() is the book-coordinate name for poisson_zero_series(); "
        "it will be removed. Use morie.fn.poisson_zero_series() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
