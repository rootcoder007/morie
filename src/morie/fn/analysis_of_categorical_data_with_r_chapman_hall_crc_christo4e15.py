"""Deprecated alias for :func:`morie.fn.poisson_rate_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisson_rate_mean import poisson_rate_mean as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_15"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_15(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.poisson_rate_mean` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_15() is the book-coordinate name for poisson_rate_mean(); "
        "it will be removed. Use morie.fn.poisson_rate_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
