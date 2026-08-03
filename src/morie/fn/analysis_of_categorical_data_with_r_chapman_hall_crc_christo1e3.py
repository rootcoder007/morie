"""Deprecated alias for :func:`morie.fn.mle_variance_pi`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .mle_variance_pi import mle_variance_pi as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_3"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.mle_variance_pi` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_3() is the book-coordinate name for mle_variance_pi(); "
        "it will be removed. Use morie.fn.mle_variance_pi() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
