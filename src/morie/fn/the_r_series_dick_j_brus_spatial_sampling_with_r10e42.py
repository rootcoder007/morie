"""Deprecated alias for :func:`morie.fn.mc_variance_via_residuals`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .mc_variance_via_residuals import mc_variance_via_residuals as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_42"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_42(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.mc_variance_via_residuals` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_42() is the book-coordinate name for mc_variance_via_residuals(); "
        "it will be removed. Use morie.fn.mc_variance_via_residuals() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
