"""Deprecated alias for :func:`morie.fn.wilson_interval`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .wilson_interval import wilson_interval as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_4"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.wilson_interval` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_4() is the book-coordinate name for wilson_interval(); "
        "it will be removed. Use morie.fn.wilson_interval() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
