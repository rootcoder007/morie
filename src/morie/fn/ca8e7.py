"""Deprecated alias for :func:`morie.fn.r2_from_f2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .r2_from_f2 import r2_from_f2 as _impl

__all__ = ["ca_chapter_8_equation_7"]


def ca_chapter_8_equation_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.r2_from_f2` instead."""
    warnings.warn(
        "ca_chapter_8_equation_7() is the book-coordinate name for r2_from_f2(); "
        "it will be removed. Use morie.fn.r2_from_f2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
