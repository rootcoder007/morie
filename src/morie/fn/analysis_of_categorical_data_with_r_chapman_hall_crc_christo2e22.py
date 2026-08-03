"""Deprecated alias for :func:`morie.fn.interaction_logit`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .interaction_logit import interaction_logit as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_22"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_22(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.interaction_logit` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_22() is the book-coordinate name for interaction_logit(); "
        "it will be removed. Use morie.fn.interaction_logit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
