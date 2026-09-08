"""Deprecated alias for :func:`morie.fn.rr_from_or`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .rr_from_or import rr_from_or as _impl

__all__ = ["ca_chapter_11_equation_29"]


def ca_chapter_11_equation_29(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.rr_from_or` instead."""
    warnings.warn(
        "ca_chapter_11_equation_29() is the book-coordinate name for rr_from_or(); "
        "it will be removed. Use morie.fn.rr_from_or() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
