"""Deprecated alias for :func:`morie.fn.trend_weights`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .trend_weights import trend_weights as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_15_equation_4"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_15_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.trend_weights` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_15_equation_4() is the book-coordinate name for trend_weights(); "
        "it will be removed. Use morie.fn.trend_weights() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
