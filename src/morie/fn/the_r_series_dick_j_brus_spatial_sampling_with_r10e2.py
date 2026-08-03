"""Deprecated alias for :func:`morie.fn.difference_estimator`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .difference_estimator import difference_estimator as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.difference_estimator` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_2() is the book-coordinate name for difference_estimator(); "
        "it will be removed. Use morie.fn.difference_estimator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
