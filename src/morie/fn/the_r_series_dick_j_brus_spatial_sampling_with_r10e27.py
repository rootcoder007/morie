"""Deprecated alias for :func:`morie.fn.ratio_g_weight`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ratio_g_weight import ratio_g_weight as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_27"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_27(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.ratio_g_weight` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_27() is the book-coordinate name for ratio_g_weight(); "
        "it will be removed. Use morie.fn.ratio_g_weight() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
