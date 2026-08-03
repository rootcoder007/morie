"""Deprecated alias for :func:`morie.fn.n_for_proportion_length`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .n_for_proportion_length import n_for_proportion_length as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_11"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_11(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.n_for_proportion_length` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_11() is the book-coordinate name for n_for_proportion_length(); "
        "it will be removed. Use morie.fn.n_for_proportion_length() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
