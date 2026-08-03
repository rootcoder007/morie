"""Deprecated alias for :func:`morie.fn.polr_parameterization`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .polr_parameterization import polr_parameterization as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_13"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_13(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.polr_parameterization` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_13() is the book-coordinate name for polr_parameterization(); "
        "it will be removed. Use morie.fn.polr_parameterization() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
