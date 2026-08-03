"""Deprecated alias for :func:`morie.fn.posterior_kernel_regression`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .posterior_kernel_regression import posterior_kernel_regression as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_25"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_25(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.posterior_kernel_regression` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_25() is the book-coordinate name for posterior_kernel_regression(); "
        "it will be removed. Use morie.fn.posterior_kernel_regression() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
