"""Deprecated alias for :func:`morie.fn.true_confidence_level`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .true_confidence_level import true_confidence_level as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_6"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_6(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.true_confidence_level` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_6() is the book-coordinate name for true_confidence_level(); "
        "it will be removed. Use morie.fn.true_confidence_level() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
