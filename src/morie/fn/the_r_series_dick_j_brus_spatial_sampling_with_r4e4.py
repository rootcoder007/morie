"""Deprecated alias for :func:`morie.fn.stratified_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .stratified_variance import stratified_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_4"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.stratified_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_4() is the book-coordinate name for stratified_variance(); "
        "it will be removed. Use morie.fn.stratified_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
