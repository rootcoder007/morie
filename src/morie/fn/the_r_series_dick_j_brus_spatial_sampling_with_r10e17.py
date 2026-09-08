"""Deprecated alias for :func:`morie.fn.g_weight_simple`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .g_weight_simple import g_weight_simple as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_17"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_17(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.g_weight_simple` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_17() is the book-coordinate name for g_weight_simple(); "
        "it will be removed. Use morie.fn.g_weight_simple() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
