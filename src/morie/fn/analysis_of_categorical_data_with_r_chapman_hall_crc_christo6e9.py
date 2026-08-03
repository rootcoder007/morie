"""Deprecated alias for :func:`morie.fn.survey_proportion_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .survey_proportion_variance import survey_proportion_variance as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_9"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_9(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.survey_proportion_variance` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_9() is the book-coordinate name for survey_proportion_variance(); "
        "it will be removed. Use morie.fn.survey_proportion_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
