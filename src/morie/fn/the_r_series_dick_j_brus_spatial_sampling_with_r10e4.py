"""Deprecated alias for :func:`morie.fn.gls_population_slope`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gls_population_slope import gls_population_slope as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_4"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.gls_population_slope` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_4() is the book-coordinate name for gls_population_slope(); "
        "it will be removed. Use morie.fn.gls_population_slope() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
