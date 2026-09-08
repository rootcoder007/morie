"""Deprecated alias for :func:`morie.fn.d_from_t`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .d_from_t import d_from_t as _impl

__all__ = ["ca_chapter_11_equation_5"]


def ca_chapter_11_equation_5(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.d_from_t` instead."""
    warnings.warn(
        "ca_chapter_11_equation_5() is the book-coordinate name for d_from_t(); "
        "it will be removed. Use morie.fn.d_from_t() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
