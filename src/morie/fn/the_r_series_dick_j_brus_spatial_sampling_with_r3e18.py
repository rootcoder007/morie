"""Deprecated alias for :func:`morie.fn.infinite_total`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .infinite_total import infinite_total as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_18"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_18(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.infinite_total` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_18() is the book-coordinate name for infinite_total(); "
        "it will be removed. Use morie.fn.infinite_total() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
