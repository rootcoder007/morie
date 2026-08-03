"""Deprecated alias for :func:`morie.fn.pearson_chi2_two_groups`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .pearson_chi2_two_groups import pearson_chi2_two_groups as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_7"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.pearson_chi2_two_groups` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_7() is the book-coordinate name for pearson_chi2_two_groups(); "
        "it will be removed. Use morie.fn.pearson_chi2_two_groups() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
