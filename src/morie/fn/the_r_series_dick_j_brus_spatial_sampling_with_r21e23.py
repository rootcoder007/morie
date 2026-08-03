"""Deprecated alias for :func:`morie.fn.gaussian_loglikelihood`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gaussian_loglikelihood import gaussian_loglikelihood as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_23"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_23(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.gaussian_loglikelihood` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_23() is the book-coordinate name for gaussian_loglikelihood(); "
        "it will be removed. Use morie.fn.gaussian_loglikelihood() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
