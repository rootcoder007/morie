"""Deprecated alias for :func:`morie.fn.twostage_optimal_n_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .twostage_optimal_n_variance import twostage_optimal_n_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_9"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_9(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.twostage_optimal_n_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_9() is the book-coordinate name for twostage_optimal_n_variance(); "
        "it will be removed. Use morie.fn.twostage_optimal_n_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
