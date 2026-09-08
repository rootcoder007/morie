"""Deprecated alias for :func:`morie.fn.ospats_objective`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ospats_objective import ospats_objective as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_17"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_17(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.ospats_objective` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_17() is the book-coordinate name for ospats_objective(); "
        "it will be removed. Use morie.fn.ospats_objective() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
