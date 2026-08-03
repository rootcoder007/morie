"""Deprecated alias for :func:`morie.fn.fpc_mean_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .fpc_mean_variance import fpc_mean_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_5"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_5(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.fpc_mean_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_5() is the book-coordinate name for fpc_mean_variance(); "
        "it will be removed. Use morie.fn.fpc_mean_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
