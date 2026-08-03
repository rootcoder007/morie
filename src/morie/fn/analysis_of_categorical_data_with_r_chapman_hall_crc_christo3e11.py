"""Deprecated alias for :func:`morie.fn.proportional_odds_logit`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .proportional_odds_logit import proportional_odds_logit as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_11"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_11(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.proportional_odds_logit` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_11() is the book-coordinate name for proportional_odds_logit(); "
        "it will be removed. Use morie.fn.proportional_odds_logit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
