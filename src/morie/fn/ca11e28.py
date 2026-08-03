"""Deprecated alias for :func:`morie.fn.or_from_rr`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .or_from_rr import or_from_rr as _impl

__all__ = ["ca_chapter_11_equation_28"]


def ca_chapter_11_equation_28(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.or_from_rr` instead."""
    warnings.warn(
        "ca_chapter_11_equation_28() is the book-coordinate name for or_from_rr(); "
        "it will be removed. Use morie.fn.or_from_rr() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
