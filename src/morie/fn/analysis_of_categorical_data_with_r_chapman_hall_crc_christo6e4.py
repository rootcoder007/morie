"""Deprecated alias for :func:`morie.fn.logistic_joint_probability`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .logistic_joint_probability import logistic_joint_probability as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_4"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.logistic_joint_probability` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_4() is the book-coordinate name for logistic_joint_probability(); "
        "it will be removed. Use morie.fn.logistic_joint_probability() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
