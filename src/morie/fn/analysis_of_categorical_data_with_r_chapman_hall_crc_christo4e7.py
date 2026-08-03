"""Deprecated alias for :func:`morie.fn.loglinear_odds_ratio`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .loglinear_odds_ratio import loglinear_odds_ratio as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_7"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.loglinear_odds_ratio` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_7() is the book-coordinate name for loglinear_odds_ratio(); "
        "it will be removed. Use morie.fn.loglinear_odds_ratio() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
