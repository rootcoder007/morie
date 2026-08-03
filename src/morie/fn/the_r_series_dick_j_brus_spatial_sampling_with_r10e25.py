"""Deprecated alias for :func:`morie.fn.ratio_total_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ratio_total_variance import ratio_total_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_25"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_25(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.ratio_total_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_25() is the book-coordinate name for ratio_total_variance(); "
        "it will be removed. Use morie.fn.ratio_total_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
