"""Deprecated alias for :func:`morie.fn.s2_residuals`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .s2_residuals import s2_residuals as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_8"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_8(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.s2_residuals` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_8() is the book-coordinate name for s2_residuals(); "
        "it will be removed. Use morie.fn.s2_residuals() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
