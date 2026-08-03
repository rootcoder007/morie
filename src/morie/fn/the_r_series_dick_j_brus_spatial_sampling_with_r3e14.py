"""Deprecated alias for :func:`morie.fn.si_proportion_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .si_proportion_variance import si_proportion_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_14"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_14(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.si_proportion_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_14() is the book-coordinate name for si_proportion_variance(); "
        "it will be removed. Use morie.fn.si_proportion_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
