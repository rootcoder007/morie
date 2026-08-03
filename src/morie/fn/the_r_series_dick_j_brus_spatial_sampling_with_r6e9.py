"""Deprecated alias for :func:`morie.fn.cluster_total_si`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cluster_total_si import cluster_total_si as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_6_equation_9"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_6_equation_9(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cluster_total_si` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_6_equation_9() is the book-coordinate name for cluster_total_si(); "
        "it will be removed. Use morie.fn.cluster_total_si() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
