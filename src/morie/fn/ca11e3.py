"""Deprecated alias for :func:`morie.fn.hedges_j`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .hedges_j import hedges_j as _impl

__all__ = ["ca_chapter_11_equation_3"]


def ca_chapter_11_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.hedges_j` instead."""
    warnings.warn(
        "ca_chapter_11_equation_3() is the book-coordinate name for hedges_j(); "
        "it will be removed. Use morie.fn.hedges_j() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
