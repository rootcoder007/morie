"""Deprecated alias for :func:`morie.fn.tolerance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .tolerance import tolerance as _impl

__all__ = ["ca_chapter_3_equation_1"]


def ca_chapter_3_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.tolerance` instead."""
    warnings.warn(
        "ca_chapter_3_equation_1() is the book-coordinate name for tolerance(); "
        "it will be removed. Use morie.fn.tolerance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
