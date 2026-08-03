"""Deprecated alias for :func:`morie.fn.cluster_mean_from_total`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cluster_mean_from_total import cluster_mean_from_total as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_6_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_6_equation_10(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cluster_mean_from_total` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_6_equation_10() is the book-coordinate name for cluster_mean_from_total(); "
        "it will be removed. Use morie.fn.cluster_mean_from_total() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
