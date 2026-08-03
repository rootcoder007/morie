"""Deprecated alias for :func:`morie.fn.local_mean_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .local_mean_variance import local_mean_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_9_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_9_equation_10(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.local_mean_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_9_equation_10() is the book-coordinate name for local_mean_variance(); "
        "it will be removed. Use morie.fn.local_mean_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
