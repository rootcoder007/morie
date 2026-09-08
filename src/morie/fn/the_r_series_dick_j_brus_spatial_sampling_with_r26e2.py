"""Deprecated alias for :func:`morie.fn.iid_mean_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .iid_mean_variance import iid_mean_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.iid_mean_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_2() is the book-coordinate name for iid_mean_variance(); "
        "it will be removed. Use morie.fn.iid_mean_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
