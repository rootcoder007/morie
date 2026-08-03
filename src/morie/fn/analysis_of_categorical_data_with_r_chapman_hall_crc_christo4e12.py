"""Deprecated alias for :func:`morie.fn.ordinal_score_mean_ratio`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ordinal_score_mean_ratio import ordinal_score_mean_ratio as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_12"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_12(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.ordinal_score_mean_ratio` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_12() is the book-coordinate name for ordinal_score_mean_ratio(); "
        "it will be removed. Use morie.fn.ordinal_score_mean_ratio() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
