"""Deprecated alias for :func:`morie.fn.poisson_loglik`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisson_loglik import poisson_loglik as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_3"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.poisson_loglik` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_3() is the book-coordinate name for poisson_loglik(); "
        "it will be removed. Use morie.fn.poisson_loglik() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
