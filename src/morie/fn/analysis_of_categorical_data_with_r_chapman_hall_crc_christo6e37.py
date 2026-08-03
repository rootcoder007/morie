"""Deprecated alias for :func:`morie.fn.spline_odds_ratio`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .spline_odds_ratio import spline_odds_ratio as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_37"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_37(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.spline_odds_ratio` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_37() is the book-coordinate name for spline_odds_ratio(); "
        "it will be removed. Use morie.fn.spline_odds_ratio() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
