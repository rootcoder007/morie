"""Deprecated alias for :func:`morie.fn.twostage_optimal_m`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .twostage_optimal_m import twostage_optimal_m as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_10(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.twostage_optimal_m` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_10() is the book-coordinate name for twostage_optimal_m(); "
        "it will be removed. Use morie.fn.twostage_optimal_m() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
