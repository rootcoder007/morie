"""Deprecated alias for :func:`morie.fn.estimation_adjusted_criterion`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .estimation_adjusted_criterion import estimation_adjusted_criterion as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_6"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_6(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.estimation_adjusted_criterion` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_6() is the book-coordinate name for estimation_adjusted_criterion(); "
        "it will be removed. Use morie.fn.estimation_adjusted_criterion() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
