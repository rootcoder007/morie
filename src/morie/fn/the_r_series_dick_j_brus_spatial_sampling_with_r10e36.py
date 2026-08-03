"""Deprecated alias for :func:`morie.fn.mixed_calibration_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .mixed_calibration_mean import mixed_calibration_mean as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_36"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_36(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.mixed_calibration_mean` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_36() is the book-coordinate name for mixed_calibration_mean(); "
        "it will be removed. Use morie.fn.mixed_calibration_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
