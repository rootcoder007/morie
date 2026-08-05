"""Deprecated alias for :func:`morie.fn.outmad`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .outmad import outmad as _impl

__all__ = ["wilcox_chapter_2_equation_14"]


def wilcox_chapter_2_equation_14(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.outmad` instead."""
    warnings.warn(
        "wilcox_chapter_2_equation_14() is the book-coordinate name for outmad(); "
        "it will be removed. Use morie.fn.outmad() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "wilcox2e14: deprecated alias for outmad"
