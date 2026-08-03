"""Deprecated alias for :func:`morie.fn.expected_squared_distance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .expected_squared_distance import expected_squared_distance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_15"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_15(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.expected_squared_distance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_15() is the book-coordinate name for expected_squared_distance(); "
        "it will be removed. Use morie.fn.expected_squared_distance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
