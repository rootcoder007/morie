"""Deprecated alias for :func:`morie.fn.logit_form`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .logit_form import logit_form as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_3"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.logit_form` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_3() is the book-coordinate name for logit_form(); "
        "it will be removed. Use morie.fn.logit_form() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
