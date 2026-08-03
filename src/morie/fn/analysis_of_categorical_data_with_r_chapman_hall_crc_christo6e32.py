"""Deprecated alias for :func:`morie.fn.group_testing_logit`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .group_testing_logit import group_testing_logit as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_32"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_32(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.group_testing_logit` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_32() is the book-coordinate name for group_testing_logit(); "
        "it will be removed. Use morie.fn.group_testing_logit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
