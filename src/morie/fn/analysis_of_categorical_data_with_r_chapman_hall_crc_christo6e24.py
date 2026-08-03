"""Deprecated alias for :func:`morie.fn.bayes_estimate_binomial`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bayes_estimate_binomial import bayes_estimate_binomial as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_24"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_24(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.bayes_estimate_binomial` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_24() is the book-coordinate name for bayes_estimate_binomial(); "
        "it will be removed. Use morie.fn.bayes_estimate_binomial() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
