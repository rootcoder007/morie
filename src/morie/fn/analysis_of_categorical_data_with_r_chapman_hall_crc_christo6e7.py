"""Deprecated alias for :func:`morie.fn.weighted_category_total`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .weighted_category_total import weighted_category_total as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_7"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.weighted_category_total` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_7() is the book-coordinate name for weighted_category_total(); "
        "it will be removed. Use morie.fn.weighted_category_total() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
