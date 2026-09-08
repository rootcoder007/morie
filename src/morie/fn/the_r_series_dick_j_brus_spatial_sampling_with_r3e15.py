"""Deprecated alias for :func:`morie.fn.confidence_interval`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .confidence_interval import confidence_interval as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_15"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_15(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.confidence_interval` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_15() is the book-coordinate name for confidence_interval(); "
        "it will be removed. Use morie.fn.confidence_interval() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
