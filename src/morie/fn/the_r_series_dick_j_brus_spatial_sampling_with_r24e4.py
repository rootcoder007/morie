"""Deprecated alias for :func:`morie.fn.augmented_kriging_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .augmented_kriging_variance import augmented_kriging_variance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_4"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.augmented_kriging_variance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_4() is the book-coordinate name for augmented_kriging_variance(); "
        "it will be removed. Use morie.fn.augmented_kriging_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
