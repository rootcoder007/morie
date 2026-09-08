"""Deprecated alias for :func:`morie.fn.bayes_rule`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bayes_rule import bayes_rule as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_22"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_22(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.bayes_rule` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_22() is the book-coordinate name for bayes_rule(); "
        "it will be removed. Use morie.fn.bayes_rule() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
