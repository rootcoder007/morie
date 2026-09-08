"""Deprecated alias for :func:`morie.fn.linear_model_prediction`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .linear_model_prediction import linear_model_prediction as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_16_equation_1"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_16_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.linear_model_prediction` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_16_equation_1() is the book-coordinate name for linear_model_prediction(); "
        "it will be removed. Use morie.fn.linear_model_prediction() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
