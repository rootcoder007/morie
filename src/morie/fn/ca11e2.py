"""Deprecated alias for :func:`morie.fn.pooled_sd`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .pooled_sd import pooled_sd as _impl

__all__ = ["ca_chapter_11_equation_2"]


def ca_chapter_11_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.pooled_sd` instead."""
    warnings.warn(
        "ca_chapter_11_equation_2() is the book-coordinate name for pooled_sd(); "
        "it will be removed. Use morie.fn.pooled_sd() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
