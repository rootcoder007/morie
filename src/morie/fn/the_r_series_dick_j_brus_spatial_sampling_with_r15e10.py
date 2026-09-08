"""Deprecated alias for :func:`morie.fn.gls_estimator`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gls_estimator import gls_estimator as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_15_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_15_equation_10(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.gls_estimator` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_15_equation_10() is the book-coordinate name for gls_estimator(); "
        "it will be removed. Use morie.fn.gls_estimator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
