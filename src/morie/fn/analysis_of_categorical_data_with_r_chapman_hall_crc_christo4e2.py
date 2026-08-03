"""Deprecated alias for :func:`morie.fn.poisson_log_link`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisson_log_link import poisson_log_link as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_2"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.poisson_log_link` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_2() is the book-coordinate name for poisson_log_link(); "
        "it will be removed. Use morie.fn.poisson_log_link() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
