"""Deprecated alias for :func:`morie.fn.outms`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .outms import outms as _impl

__all__ = ["wilcox_chapter_2_equation_13"]


def wilcox_chapter_2_equation_13(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.outms` instead."""
    warnings.warn(
        "wilcox_chapter_2_equation_13() is the book-coordinate name for outms(); "
        "it will be removed. Use morie.fn.outms() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "wilcox2e13: deprecated alias for outms"
