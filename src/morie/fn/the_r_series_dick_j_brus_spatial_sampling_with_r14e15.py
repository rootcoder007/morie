"""Deprecated alias for :func:`morie.fn.small_area_mb_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .small_area_mb_mean import small_area_mb_mean as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_14_equation_15"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_14_equation_15(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.small_area_mb_mean` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_14_equation_15() is the book-coordinate name for small_area_mb_mean(); "
        "it will be removed. Use morie.fn.small_area_mb_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
