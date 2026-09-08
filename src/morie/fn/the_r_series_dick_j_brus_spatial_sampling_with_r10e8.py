"""Deprecated alias for :func:`morie.fn.regression_estimator_general`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .regression_estimator_general import regression_estimator_general as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_8"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_8(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.regression_estimator_general` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_8() is the book-coordinate name for regression_estimator_general(); "
        "it will be removed. Use morie.fn.regression_estimator_general() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
