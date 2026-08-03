"""Deprecated alias for :func:`morie.fn.contingency_pmf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .contingency_pmf import contingency_pmf as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_2"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.contingency_pmf` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_2() is the book-coordinate name for contingency_pmf(); "
        "it will be removed. Use morie.fn.contingency_pmf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
