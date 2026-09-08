"""Deprecated alias for :func:`morie.fn.twostage_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .twostage_mean import twostage_mean as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.twostage_mean` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_2() is the book-coordinate name for twostage_mean(); "
        "it will be removed. Use morie.fn.twostage_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
