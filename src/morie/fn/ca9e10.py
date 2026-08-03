"""Deprecated alias for :func:`morie.fn.t_paired`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .t_paired import t_paired as _impl

__all__ = ["ca_chapter_9_equation_10"]


def ca_chapter_9_equation_10(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.t_paired` instead."""
    warnings.warn(
        "ca_chapter_9_equation_10() is the book-coordinate name for t_paired(); "
        "it will be removed. Use morie.fn.t_paired() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
