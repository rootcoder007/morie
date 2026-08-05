"""Deprecated alias for :func:`morie.fn.idealf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .idealf import idealf as _impl

__all__ = ["wilcox_chapter_2_equation_7"]


def wilcox_chapter_2_equation_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.idealf` instead."""
    warnings.warn(
        "wilcox_chapter_2_equation_7() is the book-coordinate name for idealf(); "
        "it will be removed. Use morie.fn.idealf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "wilcox2e7: deprecated alias for idealf"
