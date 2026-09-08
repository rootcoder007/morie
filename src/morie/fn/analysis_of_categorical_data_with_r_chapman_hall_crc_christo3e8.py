"""Deprecated alias for :func:`morie.fn.pi_j_wald_interval`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .pi_j_wald_interval import pi_j_wald_interval as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_8"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_8(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.pi_j_wald_interval` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_8() is the book-coordinate name for pi_j_wald_interval(); "
        "it will be removed. Use morie.fn.pi_j_wald_interval() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
