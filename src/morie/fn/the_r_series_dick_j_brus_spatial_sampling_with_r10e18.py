"""Deprecated alias for :func:`morie.fn.g_weighted_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .g_weighted_variance import g_weighted_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_18"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_18(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.g_weighted_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_18() is the book-coordinate name for g_weighted_variance(); "
        "it will be removed. Use morie.fn.g_weighted_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
