"""Deprecated alias for :func:`morie.fn.twostage_variance_components`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .twostage_variance_components import twostage_variance_components as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_3"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.twostage_variance_components` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_3() is the book-coordinate name for twostage_variance_components(); "
        "it will be removed. Use morie.fn.twostage_variance_components() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
