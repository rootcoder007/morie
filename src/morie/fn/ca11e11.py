"""Deprecated alias for :func:`morie.fn.se_log_or`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .se_log_or import se_log_or as _impl

__all__ = ["ca_chapter_11_equation_11"]


def ca_chapter_11_equation_11(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.se_log_or` instead."""
    warnings.warn(
        "ca_chapter_11_equation_11() is the book-coordinate name for se_log_or(); "
        "it will be removed. Use morie.fn.se_log_or() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
