"""Deprecated alias for :func:`morie.fn.classification_indicator`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .classification_indicator import classification_indicator as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_25_equation_8"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_25_equation_8(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.classification_indicator` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_25_equation_8() is the book-coordinate name for classification_indicator(); "
        "it will be removed. Use morie.fn.classification_indicator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
