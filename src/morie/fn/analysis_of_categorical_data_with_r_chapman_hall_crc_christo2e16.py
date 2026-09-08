"""Deprecated alias for :func:`morie.fn.linear_predictor_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .linear_predictor_variance import linear_predictor_variance as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_16"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_16(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.linear_predictor_variance` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_16() is the book-coordinate name for linear_predictor_variance(); "
        "it will be removed. Use morie.fn.linear_predictor_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
