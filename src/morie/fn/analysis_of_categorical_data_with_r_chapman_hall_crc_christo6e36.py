"""Deprecated alias for :func:`morie.fn.truncated_power_spline`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .truncated_power_spline import truncated_power_spline as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_36"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_36(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.truncated_power_spline` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_36() is the book-coordinate name for truncated_power_spline(); "
        "it will be removed. Use morie.fn.truncated_power_spline() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
