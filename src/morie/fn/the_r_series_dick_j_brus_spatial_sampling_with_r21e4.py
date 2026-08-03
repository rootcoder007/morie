"""Deprecated alias for :func:`morie.fn.kriging_weights_covariance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .kriging_weights_covariance import kriging_weights_covariance as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_4"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.kriging_weights_covariance` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_4() is the book-coordinate name for kriging_weights_covariance(); "
        "it will be removed. Use morie.fn.kriging_weights_covariance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
