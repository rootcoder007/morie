"""Deprecated alias for :func:`morie.fn.loglinear_saturated_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .loglinear_saturated_mean import loglinear_saturated_mean as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_6"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_6(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.loglinear_saturated_mean` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_6() is the book-coordinate name for loglinear_saturated_mean(); "
        "it will be removed. Use morie.fn.loglinear_saturated_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
