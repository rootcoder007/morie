"""Deprecated alias for :func:`morie.fn.poisson_offset_predict`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .poisson_offset_predict import poisson_offset_predict as _impl

__all__ = ["ca_chapter_6_equation_7"]


def ca_chapter_6_equation_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.poisson_offset_predict` instead."""
    warnings.warn(
        "ca_chapter_6_equation_7() is the book-coordinate name for poisson_offset_predict(); "
        "it will be removed. Use morie.fn.poisson_offset_predict() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
