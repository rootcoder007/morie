"""Deprecated alias for :func:`morie.fn.mixed_calibration_si`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .mixed_calibration_si import mixed_calibration_si as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_40"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_40(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.mixed_calibration_si` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_40() is the book-coordinate name for mixed_calibration_si(); "
        "it will be removed. Use morie.fn.mixed_calibration_si() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
