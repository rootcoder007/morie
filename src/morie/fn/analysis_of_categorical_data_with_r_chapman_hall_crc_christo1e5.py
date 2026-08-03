"""Deprecated alias for :func:`morie.fn.beta_pdf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .beta_pdf import beta_pdf as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_5"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_5(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.beta_pdf` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_5() is the book-coordinate name for beta_pdf(); "
        "it will be removed. Use morie.fn.beta_pdf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
