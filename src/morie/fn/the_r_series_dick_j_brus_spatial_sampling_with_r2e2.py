"""Deprecated alias for :func:`morie.fn.ht_total`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ht_total import ht_total as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.ht_total` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_2() is the book-coordinate name for ht_total(); "
        "it will be removed. Use morie.fn.ht_total() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
