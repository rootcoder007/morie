"""Deprecated alias for :func:`morie.fn.twostage_total_si`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .twostage_total_si import twostage_total_si as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_13"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_13(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.twostage_total_si` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_13() is the book-coordinate name for twostage_total_si(); "
        "it will be removed. Use morie.fn.twostage_total_si() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
