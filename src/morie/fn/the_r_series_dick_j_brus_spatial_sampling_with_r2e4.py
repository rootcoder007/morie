"""Deprecated alias for :func:`morie.fn.ht_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ht_mean import ht_mean as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_4"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.ht_mean` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_4() is the book-coordinate name for ht_mean(); "
        "it will be removed. Use morie.fn.ht_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
