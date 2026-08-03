"""Deprecated alias for :func:`morie.fn.autocorrelated_mean_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .autocorrelated_mean_variance import autocorrelated_mean_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_3"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.autocorrelated_mean_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_3() is the book-coordinate name for autocorrelated_mean_variance(); "
        "it will be removed. Use morie.fn.autocorrelated_mean_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
