"""Deprecated alias for :func:`morie.fn.category_prob_from_cumulative`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .category_prob_from_cumulative import category_prob_from_cumulative as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_12"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_12(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.category_prob_from_cumulative` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_12() is the book-coordinate name for category_prob_from_cumulative(); "
        "it will be removed. Use morie.fn.category_prob_from_cumulative() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
