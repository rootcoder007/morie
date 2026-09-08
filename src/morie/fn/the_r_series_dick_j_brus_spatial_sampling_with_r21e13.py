"""Deprecated alias for :func:`morie.fn.exponential_semivariogram`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .exponential_semivariogram import exponential_semivariogram as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_13"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_13(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.exponential_semivariogram` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_13() is the book-coordinate name for exponential_semivariogram(); "
        "it will be removed. Use morie.fn.exponential_semivariogram() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
