"""Deprecated alias for :func:`morie.fn.three_mrcv_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .three_mrcv_mean import three_mrcv_mean as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_16"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_16(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.three_mrcv_mean` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_16() is the book-coordinate name for three_mrcv_mean(); "
        "it will be removed. Use morie.fn.three_mrcv_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
