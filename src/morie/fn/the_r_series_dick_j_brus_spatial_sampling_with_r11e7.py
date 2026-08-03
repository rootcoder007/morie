"""Deprecated alias for :func:`morie.fn.twophase_regression_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .twophase_regression_variance import twophase_regression_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_7"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.twophase_regression_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_7() is the book-coordinate name for twophase_regression_variance(); "
        "it will be removed. Use morie.fn.twophase_regression_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
