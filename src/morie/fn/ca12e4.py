"""Deprecated alias for :func:`morie.fn.spatial_lag_reduced_form`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .spatial_lag_reduced_form import spatial_lag_reduced_form as _impl

__all__ = ["ca_chapter_12_equation_4"]


def ca_chapter_12_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.spatial_lag_reduced_form` instead."""
    warnings.warn(
        "ca_chapter_12_equation_4() is the book-coordinate name for spatial_lag_reduced_form(); "
        "it will be removed. Use morie.fn.spatial_lag_reduced_form() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
