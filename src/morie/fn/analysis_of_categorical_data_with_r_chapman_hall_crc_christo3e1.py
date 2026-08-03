"""Deprecated alias for :func:`morie.fn.multinomial_pmf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .multinomial_pmf import multinomial_pmf as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_1"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.multinomial_pmf` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_1() is the book-coordinate name for multinomial_pmf(); "
        "it will be removed. Use morie.fn.multinomial_pmf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
