"""Deprecated alias for :func:`morie.fn.twostage_total_variance_pps`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .twostage_total_variance_pps import twostage_total_variance_pps as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_12"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_12(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.twostage_total_variance_pps` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_12() is the book-coordinate name for twostage_total_variance_pps(); "
        "it will be removed. Use morie.fn.twostage_total_variance_pps() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
