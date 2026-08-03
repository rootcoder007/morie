"""Deprecated alias for :func:`morie.fn.bernoulli_likelihood`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bernoulli_likelihood import bernoulli_likelihood as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_1"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.bernoulli_likelihood` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_1() is the book-coordinate name for bernoulli_likelihood(); "
        "it will be removed. Use morie.fn.bernoulli_likelihood() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
