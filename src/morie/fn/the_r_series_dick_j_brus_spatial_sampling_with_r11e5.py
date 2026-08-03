"""Deprecated alias for :func:`morie.fn.twophase_stratified_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .twophase_stratified_variance import twophase_stratified_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_5"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_5(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.twophase_stratified_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_5() is the book-coordinate name for twophase_stratified_variance(); "
        "it will be removed. Use morie.fn.twophase_stratified_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
